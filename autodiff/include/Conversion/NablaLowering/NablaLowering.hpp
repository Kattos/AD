#ifndef AD_CONVERSION_NABLALOWERING_HPP
#define AD_CONVERSION_NABLALOWERING_HPP

#include "Dialect/Nabla/IR/Nabla.hpp"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace autodiff {

#define GEN_PASS_DECL_NABLALOWERING
#define GEN_PASS_DEF_NABLALOWERING
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createNablaLowering();

}  // namespace autodiff
}  // namespace mlir

#endif  // AD_CONVERSION_NABLALOWERING_HPP
