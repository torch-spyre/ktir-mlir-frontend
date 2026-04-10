# Kernel Tile IR (KTIR) Frontend (ktir_mlir_frontend)

KTIR is a tile-based, block-structured intermediate representation (IR) designed to express programs targeting multi-core accelerator architectures. It embodies a data-parallel abstraction, where the accelerator contains multiple cores, with each core comprising a compute engine and an on-chip scratchpad memory associated with the compute unit. The cores are attached together through an on-chip interconnect fabric, which also interfaces with one or more off-chip memory banks.

KTIR programs allow tensors to be placed in a distributed fashion across multiple memory elements (on-chip and/or off-chip memory). In the same vein, compute operations can be split into compute tiles and assigned to the compute engine within each core. KTIR allows each compute tile to have global view of all on-chip and off-chip memory elements via the on-chip interconnect fabric.

KTIR is built to represent complex computation kernels (e.g., attention in LLMs) mapped onto multi-core accelerators, going beyond a single operation (e.g., tensor Add). Kernels span multiple operations that are composed sequentially, interleaved with structured control-flow such as loops, conditionals, and multi-stage pipelines. KTIR includes constructs to capture kernel execution semantics including dependencies, synchronization points, and tensor liveliness. Intermediate tensors within the kernel could be placed on on-chip memory elements and reused across producer-consumer operations. With each compute tile having global access to all memory elements, individual compute operations can be flexibly tiled along different dimensions.

More details about KTIR can be found in the RFC: [KTIR (KernelTileIR)](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)


## Dialect Overview

The `ktdp` dialect models tile-based, data-parallel kernels targeting multi-core accelerator architectures (e.g., Spyre). It provides:

- **Memory views**: `construct_memory_view`, `construct_distributed_memory_view`
- **Access tiles**: `construct_access_tile` (direct), `construct_indirect_access_tile` (gather/scatter)
- **Data movement**: `load`, `store` (driven by access tiles)
- **Tile identity**: `get_compute_tile_id`
- **Runtime args**: `runtime_arg` type + `runtime_arg_extract` op

Custom types: `!ktdp.access_tile<NxMxindex>`, `!ktdp.runtime_arg<type, granularity=N, upperbound=M>`

Memory spaces: `#ktdp.spyre_memory_space<HBM>`, `<LX, core=7>`, `<L0>`, `<unspecified>`

## How to Build

### Prerequisites
- LLVM/MLIR built
- CMake >= 3.20, C++17 compiler

### Build
```bash
cmake -S . -B build -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
cmake --build build -j$(nproc)
```

## Running Tests

### Via CMake

From the build directory:

```bash
cmake --build <build-dir> --target check-ktir
```

### Via LIT directly

To run all tests:

```bash
llvm-lit -sv test/Ktdp/
```

To run a single test file:

```bash
llvm-lit -sv test/Ktdp/dummy.mlir
```

`llvm-lit` is located in your LLVM install's `bin/` directory.

## Use ktir-opt

```bash
build/bin/ktir-opt your_file.mlir
build/bin/ktir-opt --show-dialects
build/bin/ktir-opt --verify-roundtrip file.mlir
```

## Project Structure

```
include/
  Ktdp/                    # TableGen definitions + C++ headers
lib/
  Ktdp/                    # Dialect implementation
tools/
  ktir-opt/                # Optimizer driver tool
test/
  Ktdp/                    # 17 LIT test files
```
