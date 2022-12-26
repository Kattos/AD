#ifndef AD_CONVERSION_H
#define AD_CONVERSION_H

#include "Conversion/ADToCore/ADToCore.hpp"
#include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"
#include "Conversion/GradToCore/GradToCore.hpp"
// #include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.hpp.inc"

// namespace mlir::autodiff {
// inline void registerTosaConversionPasses() {
//   registerPass(
//       []() -> std::unique_ptr<Pass> { return tosa::createTosaToLinalg(); });
// }
// }  // namespace mlir::autodiff

#endif  // AD_CONVERSION_H
