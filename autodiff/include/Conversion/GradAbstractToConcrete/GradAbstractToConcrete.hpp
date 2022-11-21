#ifndef AD_CONVERSION_GRADABSTRACTTOCONCRETE_H
#define AD_CONVERSION_GRADABSTRACTTOCONCRETE_H

#include "Dialect/Grad/IR/Grad.hpp"
#include "Dialect/Grad/IR/GradDialect.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace autodiff {

#define GEN_PASS_DECL_GRADABSTRACTTOCONCRETE
#define GEN_PASS_DEF_GRADABSTRACTTOCONCRETE
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createGradAbstractToConcrete();

}  // namespace autodiff
}  // namespace mlir

#endif  // AD_CONVERSION_GRADABSTRACTTOCONCRETE_H
