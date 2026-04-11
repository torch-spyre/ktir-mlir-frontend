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

from mlir_ktdp.ir import Context, Module
from mlir_ktdp.dialects import ktdp_nanobind as ktdp_d


SIMPLE_MODULE = """\
module {
    func.func @example(%a: i32, %b: i32) -> i32 {
        %0 = arith.addi %a, %b : i32
        func.return %0 : i32
    }
}
"""


def test_dialect_registration():
    with Context() as ctx:
        ktdp_d.register_dialects(ctx)


def test_module_parse():
    with Context() as ctx:
        ktdp_d.register_dialects(ctx)
        module = Module.parse(SIMPLE_MODULE, ctx)
        assert module is not None


def test_walk_operations():
    with Context() as ctx:
        ktdp_d.register_dialects(ctx)
        module = Module.parse(SIMPLE_MODULE, ctx)
        ops = list(module.body.operations)
        assert len(ops) == 1
        assert ops[0].name == "func.func"

        # Walk into the func body
        inner_ops = []
        for region in ops[0].regions:
            for block in region.blocks:
                for op in block.operations:
                    inner_ops.append(op.name)
        assert "arith.addi" in inner_ops
        assert "func.return" in inner_ops
