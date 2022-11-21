#ifndef AD_H
#define AD_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace OpTrait {

namespace impl {
LogicalResult verifyResultsAreScalarTensorLike(Operation* op);
LogicalResult verifySameToAndResultType(Operation* op);
}  // namespace impl

template <typename ConcreteType>
class ResultsAreScalarTensorLike
    : public TraitBase<ConcreteType, ResultsAreScalarTensorLike> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return impl::verifyResultsAreScalarTensorLike(op);
  }
};

template <typename ConcreteType>
class SameToAndResultType
    : public TraitBase<ConcreteType, SameToAndResultType> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return impl::verifySameToAndResultType(op);
  }
};

}  // namespace OpTrait
}  // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/AD/IR/AD.h.inc"

#endif  // AD_H
