#ifndef GRAD_INTERFACE_H
#define GRAD_INTERFACE_H

// clang-format off
#include "mlir/IR/OpDefinition.h"
#include "Dialect/Grad/IR/GradInterface.h.inc"
// clang-format on

namespace mlir {
namespace autodiff {

void registerArithPartial(DialectRegistry& registry);
void registerMathPartial(DialectRegistry& registry);

}  // namespace autodiff
}  // namespace mlir

#endif  // GRAD_INTERFACE_H