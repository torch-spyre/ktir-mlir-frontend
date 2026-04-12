# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

