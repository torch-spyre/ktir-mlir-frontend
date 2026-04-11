# ===----------------------------------------------------------------------===//
#
#  Copyright 2026 The KTIR Authors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ===----------------------------------------------------------------------===//

"""
Utilities for inspecting KTIR modules.
"""

from contextlib import contextmanager
from mlir_ktdp.ir import Context, Module
from mlir_ktdp.dialects import ktdp_nanobind as ktdp_d


@contextmanager
def ktdp_context():
    """Context manager that creates an MLIR context with KTDP dialects registered."""
    with Context() as ctx:
        ktdp_d.register_dialects(ctx)
        yield ctx



def walk_module(source: str) -> None:
    """Parse an MLIR module and recursively walk all operations."""

    operations = []
    def _walk_op(op, depth: int) -> None:
        operations.append((op, depth))
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    _walk_op(child, depth + 1)

    with ktdp_context() as ctx:
        module = Module.parse(source, ctx)
        _walk_op(module.operation, 0)

    return operations

# TODO: move this block to tools_ktdp/__main__.py and invoke via
# `python -m tools_ktdp`. Running `python -m tools_ktdp.ir_utils` triggers a
# RuntimeWarning from runpy because the editable-install redirecting finder
# pre-registers this module in sys.modules before runpy executes it as __main__.
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Parse an MLIR module and walk its operations."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="MLIR file to parse (reads stdin if omitted)",
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    # traversal of the source file
    operations = walk_module(source)

    # display the operations
    for op, depth in operations:
        print(f"{'  ' * depth}{op.name}: {list(op.results.types)}")
