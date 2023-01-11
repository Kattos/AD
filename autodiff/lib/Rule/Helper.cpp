#include "Rule/Helper.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/CoversionUtils.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::autodiff {

using namespace tosa;

LogicalResult elementwiseMatchAndRewriteHelper(Operation *operation,
                                               PatternRewriter &rewriter,
                                               CalFn calFn) {
  auto generic = buildGeneric(operation, rewriter, calFn);

  if (!generic) {
    return failure();
  }

  rewriter.replaceOp(operation, generic->getResults());
  return success();
}

linalg::GenericOp buildGeneric(Operation *operation, PatternRewriter &rewriter,
                               CalFn calFn) {
  return buildGeneric(operation, operation->getOperands(),
                      operation->getResults(), rewriter, calFn);
}

linalg::GenericOp buildGeneric(Operation *operation, ValueRange newOperands,
                               ValueRange newResults, PatternRewriter &rewriter,
                               CalFn calFn) {
  auto loc = operation->getLoc();

  auto results = newResults;
  auto resultTy = newResults[0].getType().dyn_cast<ShapedType>();

  if (!resultTy) return nullptr;

  unsigned rank = resultTy.getRank();

  // Construct the indexing maps needed for linalg.generic ops.
  SmallVector<Type> bodyArgTypes;

  for (Value in : newOperands)
    bodyArgTypes.emplace_back(getElementTypeOrSelf(in.getType()));

  SmallVector<Type> opResultTypes;
  SmallVector<Value> emptyTensors;

  SmallVector<Value> dynDims;
  dynDims.resize(results.front().getType().cast<ShapedType>().getRank());

  for (auto arg : newOperands) {
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
    //     loc, resultTy.getShape(), resultTy.getElementType(), filteredDims));
    emptyTensors.push_back(buffer);
    opResultTypes.push_back(result.getType());
  }

  auto bodyResultTypes = llvm::to_vector<4>(llvm::map_range(
      emptyTensors, [](Value v) { return getElementTypeOrSelf(v); }));

  SmallVector<Value, 2> operands;
  SmallVector<AffineMap, 2> indexingMaps;
  indexingMaps.reserve(newOperands.size() + bodyResultTypes.size());

  // Input indexing maps may be broadcasted.
  for (Value operand : newOperands) {
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

  indexingMaps.append(newResults.size(), rewriter.getMultiDimIdentityMap(rank));

  bool didEncounterError = false;
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, opResultTypes, operands, emptyTensors, indexingMaps,
      getNParallelLoopsAttrs(rank),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value opResult =
            calFn(operation, blockArgs.take_front(operation->getNumOperands()),
                  bodyResultTypes, rewriter);
        if (!opResult) {
          didEncounterError = true;
          return;
        }
        nestedBuilder.create<linalg::YieldOp>(loc, opResult);
      });

  if (didEncounterError) return nullptr;

  return linalgOp;
}

}  // namespace mlir::autodiff