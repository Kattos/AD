#include "ADUtils.hpp"
#include "Dialect/AD/IR/AD.hpp"
#include "Pass/Autodiff/Passes.hpp"
#include "Rules.hpp"

namespace mlir::autodiff {

const StringRef REQGRAD = "requires_grad";

int64_t counter() {
  static int64_t index = 0;
  return ++index;
}

// create `placeholder` op and replace all uses of its operand
ad::PlaceholderOp setPlaceholder(OpBuilder& builder, Value value) {
  builder.setInsertionPointAfterValue(value);
  auto placeholder = createOp<ad::PlaceholderOp>(builder, value);
  value.replaceAllUsesExcept(placeholder, placeholder);
  return placeholder;
}

// set `requires_grad` for single op
void setOpAttr(OpBuilder& builder, Operation* op) {
  if (op->hasAttr(REQGRAD) || isa<func::ReturnOp>(op)) {
    return;
  }
  auto attr = builder.getI64IntegerAttr(counter());
  op->setAttr(REQGRAD, attr);
}

// set `requires_grad` recursively for all op users
void setGraphAttr(OpBuilder& builder, Operation* op) {
  setOpAttr(builder, op);
  for (auto user : op->getUsers()) {
    setGraphAttr(builder, user);
  }
}

int64_t attrToIndex(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return intAttr.getInt();
  }
  return -1;  // error code
}

std::map<int64_t, Value> cache;

void printCache() {
  for (auto item : cache) {
    llvm::outs() << item.first << " : ";
    item.second.dump();
  }
}

void backprop(OpBuilder& builder, Operation* op, Value outputGrad) {
  auto index = attrToIndex(op->getAttr(REQGRAD));

  // update gradients
  auto currentGrad = cache[index];
  auto grad = sum(builder, currentGrad, outputGrad);
  cache[index] = grad;

  for (auto operand : op->getOperands()) {
    auto operandGrad = getGradient(op, grad, operand);
    if (!operandGrad) {
      continue;
    }
    auto prevOp = getRelatedOperation(operand);
    backprop(builder, prevOp, operandGrad);
  }
}

void gradFunc(func::FuncOp func) {
  OpBuilder builder(func);

  // replace arguments use with `placeholder`
  auto arguments = func.getArguments();
  if (arguments.empty()) {
    return;
  }
  for (auto argument : arguments) {
    auto placeholder = setPlaceholder(builder, argument);

    // set requires_grad to each placeholder
    setGraphAttr(builder, placeholder);
  }

  // init cache
  func.getBody().walk([&](Operation* op) {
    if (op->hasAttr(REQGRAD)) {
      auto index = attrToIndex(op->getAttr(REQGRAD));
      auto value = getRelatedValue(op);
      builder.setInsertionPointAfterValue(value);
      cache[index] = zeros(builder, value);
    }
  });

  // evaluate gradients
  auto returnOp = &*func.rbegin()->rbegin();
  auto outputs = returnOp->getOperands();
  builder.setInsertionPoint(returnOp);

  for (auto output : outputs) {
    auto op = getRelatedOperation(output);
    if (!op || !op->hasAttr(REQGRAD)) {
      continue;
    }

    // set grad to ones
    auto one = ones(builder, output);

    // do backprop
    backprop(builder, op, one);
  }

  // update function signature
  auto symName = func.getSymName();
  auto newSymName = ("diff_" + symName).str();
  func.setSymName(newSymName);

  auto argsType = func.getArgumentTypes();
  auto newFuncType = builder.getFunctionType(argsType, argsType);
  func.setFunctionType(newFuncType);

  // update values to return
  returnOp->setOperands(arguments);
  for (size_t i = 0; i < arguments.size(); ++i) {
    // argument should have `placeholder` as its only user
    for (auto placeholder : arguments[i].getUsers()) {
      if (auto attr = placeholder->getAttr(REQGRAD)) {
        auto index = attrToIndex(attr);
        returnOp->setOperand(i, cache[index]);
        break;
      }
    }
  }
}

class ExperimentalPass : public ExperimentalPassBase<ExperimentalPass> {
  void runOnOperation() override { getOperation()->walk(gradFunc); }
};

std::unique_ptr<Pass> createADExperimentalPass() {
  return std::make_unique<ExperimentalPass>();
}

}  // namespace mlir::autodiff
