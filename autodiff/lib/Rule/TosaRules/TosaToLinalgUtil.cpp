#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/CoversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tosa;

using NestedFn =
    function_ref<Value(Operation *op, ValueRange args,
                       ArrayRef<Type> resultTypes, PatternRewriter &rewriter)>;

Value createLinalgBodyCalculationForElementwiseOp(Operation *op,
                                                  ValueRange args,
                                                  ArrayRef<Type> resultTypes,
                                                  PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto elementTy =
      op->getOperand(0).getType().cast<ShapedType>().getElementType();

  // tosa::AbsOp
  if (isa<tosa::AbsOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<math::AbsOp>(loc, resultTypes, args);

  if (isa<tosa::AbsOp>(op) && elementTy.isa<IntegerType>()) {
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementTy));
    auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                              args[0], zero);
    auto neg = rewriter.create<arith::SubIOp>(loc, zero, args[0]);
    return rewriter.create<arith::SelectOp>(loc, cmp, args[0], neg);
  }

  (void)rewriter.notifyMatchFailure(
      op, "unhandled op for linalg body calculation for elementwise op");
  return nullptr;
}

LogicalResult elementwiseMatchAndRewriteHelper(Operation *operation,
                                               PatternRewriter &rewriter,
                                               NestedFn fn) {
  auto loc = operation->getLoc();

  assert(operation->getNumResults() == 1 &&
         "All TOSA elementwise ops should only return a single result.");

  auto results = operation->getResults();
  auto resultTy = operation->getResult(0).getType().dyn_cast<ShapedType>();

  if (!resultTy)
    return rewriter.notifyMatchFailure(operation,
                                       "All results must be a shaped type");

  unsigned rank = resultTy.getRank();

  // Construct the indexing maps needed for linalg.generic ops.
  SmallVector<Type> bodyArgTypes;

  for (Value in : operation->getOperands())
    bodyArgTypes.emplace_back(getElementTypeOrSelf(in.getType()));

  SmallVector<Type> opResultTypes;
  SmallVector<Value> emptyTensors;

  SmallVector<Value> dynDims;
  dynDims.resize(results.front().getType().cast<ShapedType>().getRank());

  for (auto arg : operation->getOperands()) {
    auto operandTy = arg.getType().cast<ShapedType>();
    for (int i = 0; i < operandTy.getRank(); i++) {
      if (operandTy.isDynamicDim(i) && !dynDims[i])
        dynDims[i] = rewriter.create<tensor::DimOp>(loc, arg, i);
    }
  }

  SmallVector<Value> filteredDims = condenseValues(dynDims);

  for (auto result : results) {
    auto resultTy = result.getType().template cast<ShapedType>();
    auto bufferType =
        RankedTensorType::get(resultTy.getShape(), resultTy.getElementType());
    auto buffer = rewriter.create<bufferization::AllocTensorOp>(
        loc, bufferType, SmallVector<Value, 0>());
    // emptyTensors.push_back(rewriter.create<tensor::EmptyOp>(
    //     loc, resultTy.getShape(), resultTy.getElementType(),
    //     filteredDims));
    emptyTensors.push_back(buffer);
    opResultTypes.push_back(result.getType());
  }

  auto bodyResultTypes = llvm::to_vector<4>(llvm::map_range(
      emptyTensors, [](Value v) { return getElementTypeOrSelf(v); }));

  SmallVector<Value, 2> operands;
  SmallVector<AffineMap, 2> indexingMaps;
  indexingMaps.reserve(operation->getNumOperands() + bodyResultTypes.size());

  // Input indexing maps may be broadcasted.
  for (Value operand : operation->getOperands()) {
    ShapedType type = operand.getType().cast<ShapedType>();

    if (type.getShape() == resultTy.getShape()) {
      operands.push_back(operand);
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
      continue;
    }

    SmallVector<int64_t, 5> newShape;
    SmallVector<AffineExpr, 4> affineExprs;
    newShape.reserve(type.getRank());
    for (const auto &it : llvm::enumerate(type.getShape())) {
      if (it.value() == resultTy.getDimSize(it.index())) {
        newShape.push_back(it.value());
        affineExprs.push_back(
            mlir::getAffineDimExpr(it.index(), rewriter.getContext()));
      }
    }

    if (newShape.size() != rank) {
      operand = rewriter.create<tosa::ReshapeOp>(
          loc, RankedTensorType::get(newShape, type.getElementType()), operand,
          rewriter.getI64ArrayAttr(newShape));
    }

    operands.push_back(operand);
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/rank, /*symbolCount=*/0, affineExprs,
        rewriter.getContext()));
  }

  indexingMaps.append(operation->getNumResults(),
                      rewriter.getMultiDimIdentityMap(rank));

  bool didEncounterError = false;
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, opResultTypes, operands, emptyTensors, indexingMaps,
      getNParallelLoopsAttrs(rank),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value opResult =
            fn(operation, blockArgs.take_front(operation->getNumOperands()),
               bodyResultTypes, rewriter);
        if (!opResult) {
          didEncounterError = true;
          return;
        }
        nestedBuilder.create<linalg::YieldOp>(loc, opResult);
      });

  if (didEncounterError) return failure();

  rewriter.replaceOp(operation, linalgOp->getResults());
  return success();
}

namespace {

template <typename SrcOp>
class PointwiseConverter : public OpRewritePattern<SrcOp> {
 public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const final {
    return elementwiseMatchAndRewriteHelper(
        op, rewriter, createLinalgBodyCalculationForElementwiseOp);
  }
};

}  // namespace