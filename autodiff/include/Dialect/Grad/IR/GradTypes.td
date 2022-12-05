#ifndef GRAD_TYPES
#define GRAD_TYPES

include "mlir/IR/OpBase.td"

def Grad_Tensor4D : 4DTensorOf<[AnyInteger, AnyFloat]>;
def Grad_Tensor1D : 1DTensorOf<[AnyInteger, AnyFloat]>;

#endif // GRAD_TYPES
