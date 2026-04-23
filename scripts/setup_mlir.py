#!/usr/bin/env python3
"""Resolve and return the MLIR installation directory for ktir-mlir-frontend.

Prints MLIR_DIR to stdout; all other output goes to stderr.

Usage:
    uv sync --no-install-project          # create venv + install deps
    MLIR_DIR=$(uv run --no-project python scripts/setup_mlir.py)
    CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR" uv sync -v           # -v shows cmake/ninja output

    # Check artifact availability without downloading (exits 0 if available, 1 if not):
    uv run --no-project python scripts/setup_mlir.py --dry-run

Resolution order:
    1. --wheel flag       → install mlir_wheel, print its MLIR_DIR
    2. --hash / cmake/llvm-hash.txt → resolve artifact
       a. Cache hit       → print cached MLIR_DIR (no network)
       b. GitHub download → download artifact, cache, print MLIR_DIR
       c. Fallback        → install mlir_wheel, print its MLIR_DIR
    3. No hash file       → fall back to mlir_wheel
"""

import argparse
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import urllib.parse
import zipfile


def _err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ---------------------------------------------------------------------------
# OS / arch detection
# ---------------------------------------------------------------------------

def detect_os_arch():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        os_name = "ubuntu"
    elif system == "darwin":
        os_name = "macos"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    if machine in ("x86_64", "amd64"):
        arch = "x64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    return os_name, arch


# ---------------------------------------------------------------------------
# mlir_wheel fallback
# ---------------------------------------------------------------------------

def install_mlir_wheel():
    """pip-install mlir_wheel and return its MLIR_DIR path."""
    _err("Installing mlir_wheel from eudsl index...")
    subprocess.run(
        [
            "uv", "pip", "install", "mlir_wheel",
            "--find-links", "https://llvm.github.io/eudsl",
            "-v",
        ],
        check=True,
        stderr=sys.stderr,
    )
    result = subprocess.run(
        [
            "uv", "run", "--no-project", "python", "-c",
            "import mlir_wheel, pathlib; "
            "print(pathlib.Path(mlir_wheel.__file__).parent / 'lib/cmake/mlir')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_BASE = pathlib.Path.home() / ".cache" / "ktir-mlir"


def _mlir_dir_from_cache(artifact_name: str) -> str | None:
    """Return MLIR_DIR if the artifact is already cached, else None."""
    mlir_dir = _CACHE_BASE / artifact_name / "lib" / "cmake" / "mlir"
    if mlir_dir.is_dir():
        return str(mlir_dir)
    return None


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

class _PreservingHTTPRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        m = req.get_method()
        if code in (301, 302, 303, 307, 308) and m in ("GET", "HEAD"):
            old_host = urllib.parse.urlparse(req.full_url).hostname
            new_host = urllib.parse.urlparse(newurl).hostname
            if old_host == new_host:
                newheaders = {k: v for k, v in req.headers.items() if k.lower() != 'content-length'}
            else:
                newheaders = {}
            return urllib.request.Request(newurl,
                                         headers=newheaders,
                                         origin_req_host=req.origin_req_host,
                                         unverifiable=req.unverifiable)
        return None


def _make_request(url: str, token: str) -> urllib.request.Request:
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    return req


def query_artifact_id(token: str, repo: str, artifact_name: str) -> int | None:
    """Return the GitHub artifact ID for *artifact_name* in *repo*, or None."""
    url = (
        f"https://api.github.com/repos/{repo}/actions/artifacts"
        f"?name={artifact_name}&per_page=1"
    )
    try:
        with urllib.request.urlopen(_make_request(url, token)) as resp:
            data = json.loads(resp.read())
        artifacts = data.get("artifacts", [])
        _err(f"API response: total_count={data.get('total_count')}, artifacts found={len(artifacts)}")
        if artifacts:
            return artifacts[0]["id"]
    except urllib.error.HTTPError as exc:
        _err(f"GitHub API error querying artifact: {exc.code} {exc.reason}")
    return None


def download_and_cache(
    token: str, repo: str, artifact_id: int, artifact_name: str
) -> str:
    """Download artifact zip, extract inner .tar.gz, unpack to cache, return MLIR_DIR."""
    _CACHE_BASE.mkdir(parents=True, exist_ok=True)

    zip_url = (
        f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip"
    )
    _err(f"Downloading {artifact_name} from {repo}...")
    _err(f"URL: {zip_url}")
    _err(f"Token (first 8 chars): {token[:8]}...")

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = pathlib.Path(_tmp)
        zip_path = tmp / "artifact.zip"

        req = _make_request(zip_url, token)
        opener = urllib.request.build_opener(_PreservingHTTPRedirectHandler)
        with opener.open(req) as resp:
            total_size = int(resp.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024
            with open(zip_path, 'wb') as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = 100 * downloaded / total_size
                        mb = downloaded / (1024 * 1024)
                        _err(f"\rDownload progress: {pct:.1f}% ({mb:.1f}MB)", end="")

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        tar_files = list(tmp.glob("*.tar.gz"))
        if not tar_files:
            raise RuntimeError("No .tar.gz found inside artifact zip")
        tar_path = tar_files[0]

        _err()
        _err(f"Extracting {tar_path.name} to {_CACHE_BASE}/")
        with tarfile.open(tar_path) as tf:
            tf.extractall(_CACHE_BASE, filter="data")

    mlir_dir = _CACHE_BASE / artifact_name / "lib" / "cmake" / "mlir"
    if not mlir_dir.is_dir():
        raise RuntimeError(
            f"Expected MLIR_DIR not found after extraction: {mlir_dir}"
        )
    return str(mlir_dir)


# ---------------------------------------------------------------------------
# Repo resolution
# ---------------------------------------------------------------------------

def resolve_repo(repo_arg: str | None) -> str:
    """Return owner/repo: --repo arg → git remote origin → error."""
    if repo_arg:
        return repo_arg

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True,
        )
        url = result.stdout.strip()
        m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
        if m:
            return m.group(1)
    except subprocess.CalledProcessError:
        pass

    raise RuntimeError(
        "Cannot determine GitHub repo from git remote. "
        "Pass --repo <owner/repo> explicitly."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resolve MLIR installation directory for ktir-mlir-frontend"
    )
    parser.add_argument(
        "--hash", dest="llvm_hash",
        help="LLVM commit SHA or short hash (overrides cmake/llvm-hash.txt)",
    )
    parser.add_argument(
        "--wheel", action="store_true",
        help="Force mlir_wheel install (no artifact download, no cache)",
    )
    parser.add_argument(
        "--repo",
        help="GitHub repo (owner/repo) to search for artifacts. "
             "Defaults to the repo inferred from git remote origin.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check artifact availability without downloading or installing. "
             "Exits 0 if the artifact is available (cached or on GitHub), 1 if not.",
    )
    args = parser.parse_args()

    # ── Path 1: forced wheel ────────────────────────────────────────────────
    if args.wheel:
        if args.dry_run:
            _err("Dry run: would install mlir_wheel (no artifact download)")
            return
        print(install_mlir_wheel())
        _err("✓ mlir_wheel installed. Next: CMAKE_ARGS=\"-DMLIR_DIR=$MLIR_DIR\" uv sync -v  # builds and installs ktir against MLIR")
        return

    # ── Path 2: resolve hash ────────────────────────────────────────────────
    llvm_hash = args.llvm_hash
    if not llvm_hash:
        hash_file = pathlib.Path("cmake/llvm-hash.txt")
        if hash_file.exists():
            llvm_hash = hash_file.read_text().strip()

    if not llvm_hash:
        sys.exit(
            "Error: no hash specified.\n"
            "  cmake/llvm-hash.txt not found and --hash was not provided.\n"
            "  To use mlir_wheel instead, pass --wheel."
        )

    short_hash = llvm_hash[:8]
    os_name, arch = detect_os_arch()
    artifact_name = f"llvm-{short_hash}-{os_name}-{arch}"

    # ── Path 3: cache hit ───────────────────────────────────────────────────
    cached = _mlir_dir_from_cache(artifact_name)
    if cached:
        _err(f"Cache hit: {cached}")
        if args.dry_run:
            _err(f"✓ Dry run: artifact '{artifact_name}' available in local cache")
            return
        _err("✓ MLIR_DIR resolved. Next: CMAKE_ARGS=\"-DMLIR_DIR=$MLIR_DIR\" uv sync -v  # builds and installs ktir against MLIR")
        print(cached)
        return

    # ── Path 4: download from GitHub ────────────────────────────────────────
    token = os.environ.get("GIT_PAT") or os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit(
            f"Error: artifact '{artifact_name}' is not cached locally and no token is available.\n"
            "  Set GIT_PAT or GITHUB_TOKEN to download it from GitHub Actions.\n"
            "  To use mlir_wheel instead, pass --wheel."
        )

    try:
        repo = resolve_repo(args.repo)
    except RuntimeError as exc:
        sys.exit(
            f"Error: artifact '{artifact_name}' is not cached locally and the GitHub repo\n"
            f"  could not be determined: {exc}\n"
            "  Pass --repo <owner/repo> explicitly, or use --wheel to fall back to mlir_wheel."
        )

    _err(f"Querying artifact {artifact_name} in {repo}...")
    artifact_id = query_artifact_id(token, repo, artifact_name)
    if artifact_id is None:
        msg = (
            f"Error: artifact '{artifact_name}' not found in {repo}.\n"
            "  The artifact may not have been built yet for this hash, or it may have expired\n"
            "  (GitHub retains artifacts for 90 days). Trigger llvm-build.yml to rebuild it,\n"
            "  or use --wheel to fall back to mlir_wheel."
        )
        sys.exit(msg)

    if args.dry_run:
        _err(f"✓ Dry run: artifact '{artifact_name}' available in {repo} (id={artifact_id})")
        return

    try:
        mlir_dir = download_and_cache(token, repo, artifact_id, artifact_name)
        _err("✓ MLIR_DIR resolved. Next: CMAKE_ARGS=\"-DMLIR_DIR=$MLIR_DIR\" uv sync -v  # builds and installs ktir against MLIR")
        print(mlir_dir)
    except Exception as exc:
        sys.exit(
            f"Error: failed to download or extract artifact '{artifact_name}': {exc}\n"
            "  Check your token permissions and available disk space,\n"
            "  or use --wheel to fall back to mlir_wheel."
        )


if __name__ == "__main__":
    main()
