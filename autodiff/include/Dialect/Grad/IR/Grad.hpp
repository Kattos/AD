#ifndef GRAD_H
#define GRAD_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

namespace OpTrait {

namespace impl {

// TODO: implement this function
LogicalResult verifySameInputAndDerivativeType(Operation* op);
}  // namespace impl

template <typename ConcreteType>
class SameInputAndDerivativeType
    : public TraitBase<ConcreteType, SameInputAndDerivativeType> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return impl::verifySameInputAndDerivativeType(op);
  }
};

}  // namespace OpTrait

}  // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/Grad/IR/Grad.h.inc"

#endif  // GRAD_H
