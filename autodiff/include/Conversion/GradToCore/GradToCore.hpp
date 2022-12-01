#ifndef AD_CONVERSION_GRADTOCORE_H
#define AD_CONVERSION_GRADTOCORE_H

#include "Dialect/AD/IR/AD.hpp"
#include "Dialect/AD/IR/ADDialect.hpp"
#include "Dialect/Grad/IR/Grad.hpp"
#include "Dialect/Grad/IR/GradDialect.hpp"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::autodiff {

#define GEN_PASS_DECL_GRADTOCORE
#define GEN_PASS_DEF_GRADTOCORE
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createGradToCore();

namespace grad {
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
Value avgPool2dHelper(PatternRewriter& rewriter, Value dx);

inline void LLVM_ATTRIBUTE_UNUSED
populateWithGenerated(::mlir::RewritePatternSet& patterns);

#include "Conversion/GradToCore/GradToCore.hpp.inc"

}  // namespace core
}  // namespace grad

}  // namespace mlir::autodiff

#endif  // AD_CONVERSION_GRADTOCORE_H
