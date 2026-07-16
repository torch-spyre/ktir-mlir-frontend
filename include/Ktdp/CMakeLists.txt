add_ktir_dialect(KtdpOps ktdp)
add_ktir_doc(KtdpOps KtdpDialect Dialects/ -gen-dialect-doc -dialect=ktdp)

set(LLVM_TARGET_DEFINITIONS KtdpAttrInterfaces.td)
mlir_tablegen(KtdpAttrInterfaces.hpp.inc -gen-attr-interface-decls)
mlir_tablegen(KtdpAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_ktir_tablegen_target(KTIRKtdpAttrInterfacesIncGen)
add_ktir_doc(KtdpAttrs
  KtdpAttrInterfaces
  Dialects/
  -gen-attr-interface-docs)

set(LLVM_TARGET_DEFINITIONS KtdpAttrs.td)
mlir_tablegen(KtdpEnums.hpp.inc -gen-enum-decls)
mlir_tablegen(KtdpEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(KtdpAttrs.hpp.inc -gen-attrdef-decls -attrdefs-dialect=ktdp)
mlir_tablegen(KtdpAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ktdp)
add_ktir_tablegen_target(KTIRKtdpAttrsIncGen)
