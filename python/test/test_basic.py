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
