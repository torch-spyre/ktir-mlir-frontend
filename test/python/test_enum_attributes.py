# RUN: python %s

# Copyright 2026 The Torch-Spyre Authors.
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

"""Tests that spyreop ops take their enum attributes as generated IntEnums.

Each `EnumAttr` gets a generated builder that the op builder looks up. Without
it callers must hand-write the `#spyreop.…` assembly, and a renamed keyword
breaks them silently rather than at the API.
"""

from mlir_ktdp.ir import (
    Attribute,
    F16Type,
    InsertionPoint,
    Location,
    Module,
    RankedTensorType,
)
from mlir_ktdp.tools import ktdp_context
import mlir_ktdp.dialects.func as func
import mlir_ktdp.dialects.spyreop as spyreop

ComparePredicate = spyreop.ComparePredicate
ReductionKind = spyreop.ReductionKind
ReductionScope = spyreop.ReductionScope


# ---------------------------------------------------------------------------
# Enum member -> the keyword it must print as
# ---------------------------------------------------------------------------

COMPARE_KEYWORDS = [
    (ComparePredicate.Equal,        "equal"),
    (ComparePredicate.NotEqual,     "notequal"),
    (ComparePredicate.GreaterThan,  "greaterthan"),
    (ComparePredicate.GreaterEqual, "greaterequal"),
    (ComparePredicate.LesserThan,   "lesserthan"),
    (ComparePredicate.LesserEqual,  "lesserequal"),
]

REDUCTION_KIND_KEYWORDS = [
    (ReductionKind.Add,    "add"),
    (ReductionKind.Sub,    "sub"),
    (ReductionKind.Max,    "max"),
    (ReductionKind.Min,    "min"),
    (ReductionKind.Mul,    "mul"),
    (ReductionKind.AbsMax, "abs_max"),
]

REDUCTION_SCOPE_KEYWORDS = [
    (ReductionScope.InSlice,     "in_slice"),
    (ReductionScope.AcrossSlice, "across_slice"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

with ktdp_context(), Location.unknown():
    f16 = F16Type.get()
    row = RankedTensorType.get([64], f16)

    # A renamed keyword breaks here, not downstream of a hand-written literal.
    for enum, keyword, mnemonic in (
        [(e, k, "compare_predicate") for e, k in COMPARE_KEYWORDS]
        + [(e, k, "reduction_kind") for e, k in REDUCTION_KIND_KEYWORDS]
        + [(e, k, "reduction_scope") for e, k in REDUCTION_SCOPE_KEYWORDS]
    ):
        assert str(enum) == keyword, f"{enum!s} != {keyword}"
        assert Attribute.parse(f"#spyreop.{mnemonic}<{keyword}>")

    module = Module.create()
    with InsertionPoint(module.body):
        compare = func.FuncOp("compare", ([f16, f16], [f16] * len(COMPARE_KEYWORDS)))
        with InsertionPoint(compare.add_entry_block()):
            lhs, rhs = compare.arguments
            results = []
            for predicate, keyword in COMPARE_KEYWORDS:
                result = spyreop.compare(lhs, rhs, predicate)
                assert f"<{keyword}>" in str(result.owner), str(result.owner)

                # Reads back as what a parse of the same assembly gives.
                attribute = result.owner.opview.predicate
                assert attribute == Attribute.parse(
                    f"#spyreop.compare_predicate<{keyword}>"), str(attribute)

                # A ready-made Attribute still passes through untouched.
                passthrough = spyreop.compare(lhs, rhs, attribute)
                assert passthrough.owner.opview.predicate == attribute

                results.append(result)
            func.ReturnOp(results)

        # slice_reduction takes two enum attributes, not one.
        reduce = func.FuncOp("reduce", ([row], [row]))
        with InsertionPoint(reduce.add_entry_block()):
            operand, = reduce.arguments
            for kind, kind_keyword in REDUCTION_KIND_KEYWORDS:
                for scope, scope_keyword in REDUCTION_SCOPE_KEYWORDS:
                    op = spyreop.slice_reduction(operand, kind, scope).owner
                    assert f"<{kind_keyword}>" in str(op), str(op)
                    assert f"<{scope_keyword}>" in str(op), str(op)
            func.ReturnOp([operand])

    # Verifies, so the attributes satisfy the ops' constraints and are not
    # merely well-formed attributes.
    module.operation.verify()
