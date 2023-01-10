#ifndef LINALGEXT_OPS
#define LINALGEXT_OPS

include "mlir/IR/OpBase.td"

def LinalgExt_Dialect : Dialect {
    let name = "linalgext";
    let cppNamespace = "::mlir::autodiff::linalgext";
}

class LinalgExt_Op<string mnemonic, list<Trait> traits = []> : Op<LinalgExt_Dialect, mnemonic, traits>;

def LinalgExt_InitTensorOp : LinalgExt_Op<"init_tensor"> {
    let arguments = (ins
        I64ArrayAttr: $shape
    );

    let results = (outs
        AnyTensor: $tensor
    );

    let builders = [
        OpBuilder<(ins "ArrayRef<int64_t>": $shape, "Type": $elemtype)>
    ];

    let assemblyFormat = [{
        $shape attr-dict `:` type($tensor)
    }];

    let hasVerifier = 1;
}

#endif // LINALGEXT_OPS
