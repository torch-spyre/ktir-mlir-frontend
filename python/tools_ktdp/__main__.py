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

import argparse
import sys
from tools_ktdp.ir_utils import walk_module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse an MLIR module and walk its operations."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="MLIR file to parse (reads stdin if omitted)",
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    for op, depth in walk_module(source):
        print(f"{'  ' * depth}{op.name}: {list(op.results.types)}")


if __name__ == "__main__":
    main()
