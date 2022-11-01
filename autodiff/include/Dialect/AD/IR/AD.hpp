#ifndef AD_H
#define AD_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Dialect/AD/IR/AD.h.inc"

#endif  // AD_H
