#ifndef AD_CONVERSION_H
#define AD_CONVERSION_H

#include "Conversion/ADToCore/ADToCore.hpp"
#include "Conversion/GradAbstractToConcrete/GradAbstractToConcrete.hpp"
#include "Conversion/GradToCore/GradToCore.hpp"
#include "Conversion/LinalgExtConversion/LinalgExtConversion.hpp"

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.hpp.inc"

#endif  // AD_CONVERSION_H
