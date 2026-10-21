import os
import lit.formats
import lit.llvm

llvm_config = lit.llvm.llvm_config

config.name = "KTIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.ktir_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# Add tool directories to PATH
tool_dirs = [config.ktir_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["ktir-opt", "FileCheck"], tool_dirs)

# Quote tool paths that contain spaces so shell commands work correctly
config.substitutions = [
    (key, '"%s"' % val if val and " " in val and not val.startswith('"') else val)
    for key, val in config.substitutions
]

# Python bindings: add %python substitution and set PYTHONPATH.
if getattr(config, "enable_python_bindings", False):
    python_path = [
        os.path.join(config.ktir_obj_root, config.python_package_install_prefix, "..")
    ]
    if "PYTHONPATH" in os.environ:
        python_path += [os.environ["PYTHONPATH"]]
    llvm_config.with_environment("PYTHONPATH", python_path, append_path=True)
    # Ensure the configured Python interpreter is on PATH as 'python'.
    llvm_config.with_environment(
        "PATH",
        os.path.dirname(config.python_executable),
        append_path=True,
    )

