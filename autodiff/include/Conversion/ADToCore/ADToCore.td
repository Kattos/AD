#ifndef AD_CONVERSION_ADTOCORE
#define AD_CONVERSION_ADTOCORE

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Bufferization/IR/BufferizationOps.td"
include "Dialect/AD/IR/AD.td"

def PlaceholderToCore : Pat<
    (AD_PlaceholderOp $input),
    (replaceWithValue $input)
>;

def ScalarTensorToCore : Pat<
    (AD_ScalarTensorOp:$output $input),
    (NativeCodeCall<"buildScalarTensor($_builder, $0, $1)"> $input, $output)
>;

def TensorToTensor : Pat<
    (AD_ToTensorOp AnyTensor:$input),
    (replaceWithValue $input)
>;

def ScalarToTensor : Pat<
    (AD_ToTensorOp AnyTypeOf<[AnyInteger, AnyFloat]>:$input),
    (AD_ScalarTensorOp $input)
>;

def MemrefToTensor : Pat<
    (AD_ToTensorOp AnyMemRef:$input),
    (Bufferization_ToTensorOp $input)
>;

#endif // AD_CONVERSION_ADTOCORE
