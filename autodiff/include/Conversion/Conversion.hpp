#ifndef AD_CONVERSION_H
#define AD_CONVERSION_H

// #include "Conversion/ADToCore/ADToCore.hpp"
// #include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"
// #include "Conversion/GradToCore/GradToCore.hpp"
// #include "Conversion/LinalgExtConversion/LinalgExtConversion.hpp"

#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"
#include "Dialect/Grad/IR/Grad.hpp"
#include "Dialect/Grad/IR/GradDialect.hpp"
#include "Dialect/LinalgExt/IR/LinalgExt.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace autodiff {

std::unique_ptr<Pass> createADToCore();
std::unique_ptr<Pass> createAllocTensorToInitTensor();
std::unique_ptr<Pass> createInitTensorToAllocTensor();
std::unique_ptr<Pass> createGradAbstractToConcrete();
std::unique_ptr<Pass> createGradToCore();

namespace ad {

Value buildScalarTensor(PatternRewriter& rewriter, Value input, Value output);

inline void LLVM_ATTRIBUTE_UNUSED
populateWithGenerated(::mlir::RewritePatternSet& patterns);

#include "Conversion/ADToCore/ADToCore.hpp.inc"

}  // namespace ad

namespace grad {
namespace concrete {

Value toClamp(PatternRewriter& rewriter, Value unary);

template <typename OpTy>
Value toConcreteWithAttrs(PatternRewriter& rewriter, Value unary) {
  auto abstract = unary.getDefiningOp<grad::AbstractUnaryOp>();

  if (!abstract) {
    return nullptr;
  }

  auto resultTypes = abstract->getResultTypes();
  auto operands = abstract->getOperands();
  abstract->removeAttr("op");
  auto attrs = abstract->getAttrs();

  return rewriter.create<OpTy>(rewriter.getUnknownLoc(), resultTypes, operands,
                               attrs);
}

inline void LLVM_ATTRIBUTE_UNUSED
populateWithGenerated(::mlir::RewritePatternSet& patterns);

#include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp.inc"

}  // namespace concrete

namespace core {

Value add(PatternRewriter& rewriter, Value lhs, Value rhs);
Value mul(PatternRewriter& rewriter, Value lhs, Value rhs);
Value negate(PatternRewriter& rewriter, Value tensor);
Value exp(PatternRewriter& rewriter, Value tensor);
Value reciprocal(PatternRewriter& rewriter, Value tensor);

Value oneslike(PatternRewriter& rewriter, Value tensor);
Value broadcast(PatternRewriter& rewriter, Value from, Value to);
Value reduce(PatternRewriter& rewriter, Value from, Value to);

Value drsqrt(PatternRewriter& rewriter, Value tensor);
Value dabs(PatternRewriter& rewriter, Value tensor);
Value dGreaterEqual(PatternRewriter& rewriter, Value first, Value second);
Value intClampHelper(PatternRewriter& rewriter, Value tensor, Attribute min,
                     Attribute max);
Value floatClampHelper(PatternRewriter& rewriter, Value tensor, Attribute min,
                       Attribute max);
Value dAvgPool2d(PatternRewriter& rewriter, Value dx);

Value dConv2DInput(PatternRewriter& rewriter, Value input);
Value dConv2DBias(PatternRewriter& rewriter, Value bias);
Value dReshape(PatternRewriter& rewriter, Value dx);

inline void LLVM_ATTRIBUTE_UNUSED
populateWithGenerated(::mlir::RewritePatternSet& patterns);

#include "Conversion/GradToCore/GradToCore.hpp.inc"

}  // namespace core
}  // namespace grad

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.hpp.inc"

}  // namespace autodiff
}  // namespace mlir

#endif  // AD_CONVERSION_H
