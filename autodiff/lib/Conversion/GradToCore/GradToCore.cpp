#include "Conversion/GradToCore/GradToCore.hpp"

namespace mlir::autodiff {

class GradToCore : public impl::GradToCoreBase<GradToCore> {
  void runOnOperation() override {}
};

std::unique_ptr<Pass> createGradToCore() {
  return std::make_unique<GradToCore>();
}

}  // namespace mlir::autodiff
