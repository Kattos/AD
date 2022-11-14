# AD

AD is a set of MLIR-based tools that enables automatic diferentiation within MLIR pipelines.

**NOTE: This project is at very beginning and updates frequently**

## Getting Started

1. Build MLIR with LLVM, see [MLIR Getting Started](https://mlir.llvm.org/getting_started/)

2. Build AD

    ```sh
    git clone https://github.com/Kattos/AD
    mkdir AD/build && cd AD/build
    cmake .. -G Ninja # -DMLIR_DIR=/path/to/your/mlir
    ```

3. Take a look around

    ```sh
    autodiff-opt --help | grep "ad" # see what's new
    ```
