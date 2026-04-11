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

from mlir_ktdp.ir import Context, Module
from mlir_ktdp.dialects import ktdp_nanobind as ktdp_d


def walk_module(source: str) -> None:
    """Parse an MLIR module and print a walk of its operations."""
    with Context() as ctx:
        ktdp_d.register_dialects(ctx)
        module = Module.parse(source, ctx)
        module.dump()

        print("=== Walking Operations ===")
        for op in module.body.operations:
            print(f"top-level: {op.name}")
            for region in op.regions:
                for block in region.blocks:
                    for inner_op in block.operations:
                        print(
                            f"  {inner_op.name}: "
                            f"{list(inner_op.operation.results.types)}"
                        )


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

    walk_module(source)
