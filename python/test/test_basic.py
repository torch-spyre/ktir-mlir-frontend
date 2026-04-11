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

import pytest
from tools_ktdp import ktdp_context
from tools_ktdp.ir_utils import walk_module


# ---------------------------------------------------------------------------
# Module + expected walk pairs
# ---------------------------------------------------------------------------

SIMPLE_MODULE = """\
module {
    func.func @example(%a: i32, %b: i32) -> i32 {
        %0 = arith.addi %a, %b : i32
        func.return %0 : i32
    }
}
"""
SIMPLE_MODULE_OPS = [
    ("builtin.module", 0),
    ("func.func",      1),
    ("arith.addi",     2),
    ("func.return",    2),
]

# Flat KTDP ops: memory view, access tile, load, store — no control flow.
KTDP_BASIC_MODULE = """\
#set      = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @basic() {
    %c0     = arith.constant 0 : index
    %A_addr = arith.constant 1024 : index
    %A_view = ktdp.construct_memory_view %A_addr, sizes: [32, 64], strides: [64, 1] {
                  coordinate_set = #set,
                  memory_space   = #ktdp.spyre_memory_space<HBM>
              } : memref<32x64xf16>
    %A_tile = ktdp.construct_access_tile %A_view[%c0, %c0] {
                  access_tile_set   = #set,
                  access_tile_order = #identity
              } : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>
    %data   = ktdp.load %A_tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>
    ktdp.store %data, %A_tile : tensor<32x64xf16>, !ktdp.access_tile<32x64xindex>
    return
  }
}
"""
KTDP_BASIC_MODULE_OPS = [
    ("builtin.module",              0),
    ("func.func",                   1),
    ("arith.constant",              2),
    ("arith.constant",              2),
    ("ktdp.construct_memory_view",  2),
    ("ktdp.construct_access_tile",  2),
    ("ktdp.load",                   2),
    ("ktdp.store",                  2),
    ("func.return",                 2),
]

# Nested KTDP ops: ktdp.load / ktdp.construct_access_tile are inside an scf.for.
KTDP_CONTROL_FLOW_MODULE = """\
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_loop() {
    %c0        = arith.constant 0 : index
    %c1        = arith.constant 1 : index
    %tile_size = arith.constant 3 : index
    %A_addr    = arith.constant 1024 : index
    %id        = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index
    %A_view    = ktdp.construct_memory_view %A_addr, sizes: [96, 64], strides: [64, 1] {
                     coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
                     memory_space   = #ktdp.spyre_memory_space<HBM>
                 } : memref<96x64xf16>
    scf.for %i = %c0 to %tile_size step %c1 {
      %A_tile = ktdp.construct_access_tile %A_view[%start_row + %i, %c0] {
                    access_tile_set   = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
                    access_tile_order = #identity
                } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
      %data   = ktdp.load %A_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    }
    return
  }
}
"""
KTDP_CONTROL_FLOW_MODULE_OPS = [
    ("builtin.module",             0),
    ("func.func",                  1),
    ("arith.constant",             2),
    ("arith.constant",             2),
    ("arith.constant",             2),
    ("arith.constant",             2),
    ("ktdp.get_compute_tile_id",   2),
    ("arith.muli",                 2),
    ("ktdp.construct_memory_view", 2),
    ("scf.for",                    2),
    ("ktdp.construct_access_tile", 3),
    ("ktdp.load",                  3),
    ("scf.yield",                  3),
    ("func.return",                2),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dialect_registration():
    with ktdp_context():
        pass


@pytest.mark.parametrize("source,expected", [
    (SIMPLE_MODULE,              SIMPLE_MODULE_OPS),
    (KTDP_BASIC_MODULE,          KTDP_BASIC_MODULE_OPS),
    (KTDP_CONTROL_FLOW_MODULE,   KTDP_CONTROL_FLOW_MODULE_OPS),
])
def test_walk(source, expected):
    ops = walk_module(source)
    assert [(op.name, depth) for op, depth in ops] == expected
