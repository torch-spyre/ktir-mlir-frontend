add_ktir_dialect(Spyre spyre)
add_ktir_doc(Spyre SpyreDialect Dialects/ -gen-dialect-doc -dialect=spyre)

set(LLVM_TARGET_DEFINITIONS SpyreAttrs.td)
mlir_tablegen(SpyreAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=spyre)
mlir_tablegen(SpyreAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=spyre)
add_ktir_tablegen_target(KTIRSpyreAttrsIncGen)
add_dependencies(KTIRSpyreIncGen KTIRSpyreAttrsIncGen)
