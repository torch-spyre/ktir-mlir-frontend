# CI Overview

> The pinned LLVM artifact CI flow is adopted from
> [triton-lang/triton](https://github.com/triton-lang/triton).
> The scheduled artifact refresh and `mlir_wheel` fallback are extensions of
> that pattern.

This document describes the three CI flows for `ktir-mlir-frontend`, how they
relate to each other, and how to use them for local development.

---

## MLIR source strategy

All CI flows build the KTIR project against an MLIR installation.  There are
two sources:

| Source | When used | Stability |
|--------|-----------|-----------|
| **Custom LLVM artifact** | `cmake/llvm-hash.txt` is present | Stable — pinned to an official LLVM release tag, stored as a GitHub Actions artifact (90-day retention, refreshed every 2 months) |
| **mlir_wheel** | No hash file, or explicit `--wheel` override | Bleeding-edge — tracks LLVM `main`, individual versions expire after 30 days |

The file `cmake/llvm-hash.txt` is the single control point.  Its presence
switches all three flows to the custom artifact path automatically.

---

## Flow 1 — Standard CI (`ci.yml`)

**Triggers:** push or pull request to `main`

**What it does:**

1. Reads `cmake/llvm-hash.txt` to determine the MLIR source
2. Runs `uv sync --extra test` to install Python test dependencies (venv must
   exist before MLIR setup in case the wheel fallback needs to `pip install`)
3. Calls `scripts/setup_mlir.py` to resolve and cache the MLIR installation:
   - Default: downloads the pinned LLVM artifact from GitHub Actions
   - `--wheel`: installs `mlir_wheel` from the eudsl index (explicit opt-in only)
4. Configures and builds KTIR with CMake
5. Runs LIT tests (`check-ktir`)
6. Builds and installs the Python wheel (`uv pip install .`)
7. Runs Python tests (`pytest python/test/`)

**Normal developer workflow:** open a PR → Flow 1 runs automatically.  No
manual steps needed as long as the LLVM artifact for the pinned hash exists.

---

## Flow 2 — LLVM Build (`llvm-build.yml`)

**Triggers:**
- Push to `main` that changes `cmake/llvm-hash.txt` (hash bump)
- Scheduled: 1st of every other month at 02:00 UTC (`0 2 1 */2 *`)
- Manual: `workflow_dispatch` with an optional hash override

**What it does** depends on whether a valid artifact already exists:

### Hash bump (artifact does not exist)

1. Reads the new hash from `cmake/llvm-hash.txt`
2. Checks GitHub Actions artifacts — none found for this hash
3. Checks out `llvm-project` at the pinned commit
4. Builds LLVM/MLIR with `MLIR_ENABLE_BINDINGS_PYTHON=ON` (required for
   downstream Python wheel builds)
5. Runs `check-mlir` to validate the build
6. Packages and uploads the artifact (`retention-days: 90`)
7. Triggers Flow 1 (`ci.yml`) against the new artifact

### Scheduled refresh (artifact exists)

The scheduled run exists solely to reset the 90-day retention clock before
the artifact expires.  A full rebuild is unnecessary — the content is
identical.

1. Reads the hash from `cmake/llvm-hash.txt`
2. Checks GitHub Actions artifacts — existing artifact found
3. Downloads the artifact zip via the GitHub API
4. Re-uploads it — a new artifact entry is created with a fresh 90-day clock
5. Deletes the old artifact to prevent stale duplicates accumulating in GitHub Actions
6. Flow 1 is **not** triggered (artifact content unchanged)

The 2-month schedule gives a ~30-day buffer before the previous upload
expires, so there is always a valid artifact available for Flow 1.

### When to trigger manually

```bash
# Rebuild for the current pinned hash (e.g. after accidental artifact deletion):
gh workflow run llvm-build.yml

# Build a specific hash (overrides cmake/llvm-hash.txt):
gh workflow run llvm-build.yml -f llvm-hash=<full-40-char-sha>
```

---

## Flow 3 — Bleeding-edge (`workflow_dispatch` with mlir_wheel)

**Trigger:** manual `workflow_dispatch` on `ci.yml` with `mlir-source=mlir_wheel`

**What it does:** runs the same build + test pipeline as Flow 1 but sources
MLIR from the latest `mlir_wheel` on the eudsl index instead of the pinned
artifact.

Use this to:
- Test compatibility with the latest LLVM `main` before bumping the hash
- Quickly verify a fix without waiting for a full LLVM build

**Note:** `mlir_wheel` builds expire after 30 days and are not tied to
official LLVM release tags.  Do not rely on a specific wheel version being
available long-term.

```bash
gh workflow run ci.yml -f mlir-source=mlir_wheel
```

---

## Local development

### Set up MLIR

There are two ways to obtain an MLIR installation for local builds:

**Option A — Download the CI artifact (recommended)**

`scripts/setup_mlir.py` downloads the pre-built LLVM artifact produced by
Flow 2 (`llvm-build.yml`).  It reads `cmake/llvm-hash.txt`, checks a local
cache, and pulls from GitHub Actions if needed.  If the artifact cannot be
resolved (missing token, artifact not found, download failure), the script
exits with a clear error explaining the cause.  Pass `--wheel` to explicitly
opt in to `mlir_wheel` instead.

```bash
# Cache hit (hash unchanged since last run) — no token needed:
MLIR_DIR=$(uv run python scripts/setup_mlir.py)

# Cache miss — GIT_PAT or GITHUB_TOKEN must be set to download the artifact.
# The script resolves the repo from git remote automatically; for forks where
# the artifact lives in the upstream repo, pass --repo explicitly:
GIT_PAT=<your-token> MLIR_DIR=$(uv run python scripts/setup_mlir.py)
GIT_PAT=<your-token> MLIR_DIR=$(uv run python scripts/setup_mlir.py --repo <fork>/ktir-mlir-frontend)

# Force mlir_wheel (no token required, no cache):
MLIR_DIR=$(uv run python scripts/setup_mlir.py --wheel)
```

Artifacts are cached at `~/.cache/ktir-mlir/<artifact-name>/`.  Once cached,
subsequent calls with the same hash return immediately with no network access
and no token required.

**Option B — Build MLIR manually**

If you have built LLVM/MLIR from source yourself, skip the script entirely and
set `MLIR_DIR` directly to the `lib/cmake/mlir` directory of your build:

```bash
MLIR_DIR=/path/to/your/llvm-build/lib/cmake/mlir
```

### Build

```bash
# Install Python test dependencies
uv sync --extra test

# Build and install the Python wheel
CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR" uv pip install .

# Run tests
cmake --build build --target check-ktir   # LIT tests
uv run pytest python/test/                # Python tests
```

---

## Hash bump procedure

To adopt a new LLVM release:

1. Update `cmake/llvm-hash.txt` with the full 40-character commit SHA
2. Push to `main` (or merge a PR that changes the file)
3. Flow 2 fires automatically — builds LLVM, uploads artifact, triggers Flow 1
4. Monitor the `llvm-build` and `cmake-py-test` workflow runs

```bash
# Example: adopt llvmorg-22.1.3
echo "e9846648fd6183ee6d8cbdb4502213fcf902a211" > cmake/llvm-hash.txt
git add cmake/llvm-hash.txt
git commit -m "Bump LLVM to llvmorg-22.1.3"
git push
```
