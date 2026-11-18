// SPDX-License-Identifier: Apache-2.0
// RUN: ktir-opt %s -split-input-file -verify-diagnostics | ktir-opt --split-input-file -o /dev/null
// Element orders map relative memory coordinates to tensor coordinates.
// These tests verify the relation, not target-specific data movement.

func.func @standard_to_staggered(%mem: memref<6x2x32xf32>) {
  %tile = ktdp.construct_access_tile %mem[0, 0, 0] {
    access_tile_set = affine_set<(k,r,e) :
      (k >= 0, 5-k >= 0, r >= 0, 1-r >= 0, e >= 0, 31-e >= 0)>,
    access_tile_order = affine_map<(k,r,e) ->
      (2*(k floordiv 2)+(e mod 8) floordiv 4, r,
       16*(k mod 2)+4*(e floordiv 8)+e mod 4)>
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

func.func @staggered_to_standard(%mem: memref<2x32xf32>) {
  %tile = ktdp.construct_access_tile %mem[0, 0] {
    access_tile_set = affine_set<(k,e) :
      (k >= 0, 1-k >= 0, e >= 0, 31-e >= 0)>,
    access_tile_order = affine_map<(k,e) ->
      (e floordiv 16, 8*((e mod 16) floordiv 4)+4*k+e mod 4)>
  } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  return
}

// -----

// Symbolic relative origins are part of the relation, not added after order.
func.func @symbolic_origin(%mem: memref<?xf32>, %origin: index) {
  %tile = ktdp.construct_access_tile %mem[0] symbols(%origin) {
    access_tile_set = affine_set<(p)[s] : (p-s >= 0, 31+s-p >= 0)>,
    access_tile_order = affine_map<(p)[s] -> (p-s)>
  } : memref<?xf32> -> !ktdp.access_tile<32xindex>
  return
}

// -----

// The last partial group is a separate bijection; no implicit padding.
func.func @tail(%mem: memref<3xf32>) {
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 2-p >= 0)>,
    access_tile_order = affine_map<(p) -> (2-p)>
  } : memref<3xf32> -> !ktdp.access_tile<3xindex>
  return
}

// -----

func.func @repeated_read(%mem: memref<32xf32>) {
  // expected-error @+1 {{access_tile_order is not injective on access_tile_set}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 31-p >= 0)>,
    access_tile_order = affine_map<(p) -> (p mod 16)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  return
}

// -----

func.func @out_of_bounds(%mem: memref<32xf32>) {
  // expected-error @+1 {{access_tile_order maps outside the result shape}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 31-p >= 0)>,
    access_tile_order = affine_map<(p) -> (p+1)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  return
}

// -----

func.func @missing_elements(%mem: memref<32xf32>) {
  // expected-error @+1 {{access_tile_order does not cover the result shape}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 15-p >= 0)>,
    access_tile_order = affine_map<(p) -> (p*2)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  return
}

// -----

func.func @unpaired_packet(%mem: memref<3x32xf32>) {
  // expected-error @+1 {{access_tile_order maps outside the result shape}}
  %tile = ktdp.construct_access_tile %mem[0, 0] {
    access_tile_set = affine_set<(k,e) :
      (k >= 0, 2-k >= 0, e >= 0, 31-e >= 0)>,
    access_tile_order = affine_map<(k,e) ->
      (2*(k floordiv 2)+(e mod 8) floordiv 4,
       16*(k mod 2)+4*(e floordiv 8)+e mod 4)>
  } : memref<3x32xf32> -> !ktdp.access_tile<3x32xindex>
  return
}

// -----

func.func @unbound_symbol(%mem: memref<32xf32>) {
  // expected-error @+1 {{access_tile_order symbols must match access_tile_set bindings}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 31-p >= 0)>,
    access_tile_order = affine_map<(p)[s] -> (p+s)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  return
}

// -----

func.func @dynamic_result(%mem: memref<?xf32>, %size: index) {
  // expected-error @+1 {{element access_tile_order requires a static result shape}}
  %tile = ktdp.construct_access_tile %mem[0] symbols(%size) {
    access_tile_set = affine_set<(p)[s] : (p >= 0, s-1-p >= 0)>,
    access_tile_order = affine_map<(p)[s] -> (s-1-p)>
  } : memref<?xf32> -> !ktdp.access_tile<?xindex>
  return
}

// -----

// Equal bounding boxes do not prove coverage when the source has holes.
func.func @domain_with_holes(%mem: memref<32xf32>) {
  // expected-error @+1 {{access_tile_order does not cover the result shape}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) :
      (p >= 0, 31-p >= 0, p mod 2 == 0)>,
    access_tile_order = affine_map<(p) -> (31-p)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  return
}

// -----

// A nonrectangular source can still map bijectively onto the tensor domain.
func.func @sparse_bijection(%mem: memref<32xf32>) {
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) :
      (p >= 0, 31-p >= 0, p mod 2 == 0)>,
    access_tile_order = affine_map<(p) -> (p floordiv 2)>
  } : memref<32xf32> -> !ktdp.access_tile<16xindex>
  return
}

// -----

// Reject semi-affine sets before the Presburger constructor asserts.
func.func @semi_affine_set(%mem: memref<?xf32>, %s: index) {
  // expected-error @+1 {{element access_tile_order requires a Presburger access_tile_set}}
  %tile = ktdp.construct_access_tile %mem[0] symbols(%s) {
    access_tile_set = affine_set<(p)[s] : (p >= 0, 31-p*s >= 0)>,
    access_tile_order = affine_map<(p) -> (31-p)>
  } : memref<?xf32> -> !ktdp.access_tile<32xindex>
  return
}

// -----

func.func @semi_affine_order(%mem: memref<32xf32>, %s: index) {
  // expected-error @+1 {{element access_tile_order requires a Presburger affine map}}
  %tile = ktdp.construct_access_tile %mem[0] symbols(%s) {
    access_tile_set = affine_set<(p)[s] : (p >= 0, 31-p >= 0)>,
    access_tile_order = affine_map<(p)[s] -> (p*s)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  return
}
