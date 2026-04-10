# ===== Dialect (KtdpDialect.td) =====
set(LLVM_TARGET_DEFINITIONS KtdpDialect.td)
mlir_tablegen(KtdpDialect.hpp.inc -gen-dialect-decls -dialect=ktdp)
mlir_tablegen(KtdpDialect.cpp.inc -gen-dialect-defs  -dialect=ktdp)
add_public_tablegen_target(MLIRKtdpDialectIncGen)

# ===== Attrs (KtdpAttrs.td) =====
set(LLVM_TARGET_DEFINITIONS KtdpAttrs.td)
mlir_tablegen(KtdpAttrs.hpp.inc -gen-attrdef-decls -attrdefs-dialect=ktdp)
mlir_tablegen(KtdpAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ktdp)
mlir_tablegen(KtdpEnums.hpp.inc -gen-enum-decls)
mlir_tablegen(KtdpEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(KtdpAttrInterfaces.hpp.inc -gen-attr-interface-decls)
mlir_tablegen(KtdpAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRKtdpAttrsIncGen)

# ===== Types (KtdpTypes.td) =====
set(LLVM_TARGET_DEFINITIONS KtdpTypes.td)
mlir_tablegen(KtdpTypes.hpp.inc -gen-typedef-decls -typedefs-dialect=ktdp)
mlir_tablegen(KtdpTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ktdp)
add_public_tablegen_target(MLIRKtdpTypesIncGen)

# ===== Ops (KtdpOps.td) =====
set(LLVM_TARGET_DEFINITIONS KtdpOps.td)
mlir_tablegen(KtdpOps.hpp.inc -gen-op-decls)
mlir_tablegen(KtdpOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRKtdpOpsIncGen)

# ===== Documentation =====
set(LLVM_TARGET_DEFINITIONS KtdpDialect.td)
add_mlir_doc(KtdpDialect KtdpDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS KtdpOps.td)
add_mlir_doc(KtdpOps KtdpOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS KtdpTypes.td)
add_mlir_doc(KtdpTypes KtdpTypes Dialects/ -gen-typedef-doc)

set(LLVM_TARGET_DEFINITIONS KtdpAttrs.td)
add_mlir_doc(KtdpAttrs KtdpAttrs Dialects/ -gen-attrdef-doc)
