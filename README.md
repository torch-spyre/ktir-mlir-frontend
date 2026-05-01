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
- CMake >= 3.20, Ninja, C++17 compiler
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for Python environment)
- Python >= 3.12 (only for Python bindings)

### CMake (C++ only)

```bash
cmake -S . -B build -GNinja -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
cmake --build build -j$(nproc)
```

### CMake with Python bindings

> **Note:** The recommended way to build Python bindings is via `uv sync`, see the next section.

Python bindings are disabled by default. To enable them, add
`-DKTIR_ENABLE_PYTHON_BINDINGS=ON` and ensure `nanobind` is installed in the
active Python environment:

```bash
cmake -S . -B build -GNinja \
    -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
    -DKTIR_ENABLE_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DLLVM_EXTERNAL_LIT=$(which lit)
cmake --build build -j$(nproc)
```

The built package lands at `build/python_packages/ktdp/` — use
`PYTHONPATH=build/python_packages/ktdp` to import `mlir_ktdp`.

> **Note:** `-DLLVM_EXTERNAL_LIT` is only needed to run LIT tests. `mlir_wheel`
> does not ship `llvm-lit`, so cmake cannot find it automatically.

### pip install (scikit-build-core)

The project uses [scikit-build-core](https://scikit-build-core.readthedocs.io/) to
drive CMake and produce a wheel. First, create a venv:

```bash
uv venv --python 3.12
```

Use `scripts/setup_mlir.py` to obtain the MLIR installation, then build:

```bash
# Install Python test dependencies (pytest and lit)
uv sync --no-install-project --extra test

# Resolve MLIR.
# Skip this if you already have a local LLVM/MLIR build — just set MLIR_DIR
# directly: export MLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
#
# Otherwise, setup_mlir.py downloads the pinned artifact (cached at
# ~/.cache/ktir-mlir/) or falls back to mlir_wheel.
# A GIT_PAT or GITHUB_TOKEN is required on the first download; subsequent
# runs use the cache and need no token.
MLIR_DIR=$(uv run --no-project python scripts/setup_mlir.py)

# Force mlir_wheel (no token required, always fetches latest):
MLIR_DIR=$(uv run --no-project python scripts/setup_mlir.py --wheel)

# Build and install the Python wheel
CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR" uv pip install .
```

See [docs/ci.md](docs/ci.md) for the full local development workflow including
cache behaviour and fork setup.

## Running Tests

### LIT tests (C++ / IR)

```bash
cmake --build build --target check-ktir
```

Or directly — lit must be pointed at the **build** directory where cmake
generates `lit.site.cfg.py`:

```bash
llvm-lit -sv build/test/Ktdp/
# or, with uv (lit is installed with --extra test):
uv run lit -sv build/test/Ktdp/
```

### Python tests

```bash
uv run pytest python/test/
```

For bare cmake builds (no pip install), conftest.py adds
`build/python_packages/ktdp` to `sys.path` automatically.

## Python Tools

### ktdp-walk

Parse an MLIR file and print a recursive walk of all operations with indentation showing nesting depth:

```bash
uv run ktdp-walk path/to/file.mlir
# or from stdin:
cat file.mlir | uv run ktdp-walk
```

`ktdp-walk` is also usable as a library:

```python
from tools_ktdp import ktdp_context
from tools_ktdp.ir_utils import walk_module

# context manager
with ktdp_context() as ctx:
    ...

# walk returns [(op, depth), ...]
for op, depth in walk_module(source):
    print(f"{'  ' * depth}{op.name}")
```

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
