# SPDX-License-Identifier: GPL-2.0-or-later

"""Check Northwood's first-party source and payload policy over a Git diff."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


SOURCE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".h", ".hpp", ".metal", ".m", ".mm", ".py",
    ".rs", ".s", ".sh", ".swift",
}
PAYLOAD_EXTENSIONS = {".fw", ".ucode", ".ipsw"}
LARGE_BINARY = 1024 * 1024
PORTABLE_LICENSES = {
    "0BSD", "BSD-2-Clause", "BSD-3-Clause", "BSD-3-Clause-Clear", "BSL-1.0",
    "CC0-1.0", "GPL-2.0-or-later", "ISC", "MIT", "Unicode-3.0", "Unicode-DFS-2015",
    "Zlib", "Apache-2.0 WITH LLVM-exception",
}
CLI_LICENSES = {"Apache-2.0", "GPL-3.0-only", "GPL-3.0-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later"}
BOOT_LICENSES = {"GPL-2.0-only", "GPL-2.0-only WITH Linux-syscall-note"}
LICENSE_TEXT = {
    "GPL-2.0-or-later": "GPL-2.0-only.txt",
    "GPL-3.0-only": "GPL-3.0-or-later.txt",
    "Apache-2.0 WITH LLVM-exception": "Apache-2.0-WITH-LLVM-exception.txt",
}
ALIASES = {"GPL-2.0": "GPL-2.0-only", "GPL-2.0+": "GPL-2.0-or-later"}
TOKEN = re.compile(r"\(|\)|[A-Za-z0-9.-]+\+?")
HEADER = re.compile(r"^\s*(?://|#|/\*|\*)\s*SPDX-License-Identifier:\s*(.+?)\s*(?:\*/)?$")
PORT = re.compile(r"\b(?:ported|adapted) from\b", re.IGNORECASE)
RESTRICTED = re.compile(r"^(?:PolyForm-|CC-BY-NC|LicenseRef-)", re.IGNORECASE)


def check_expression(expression, allowed):
    """Check whether one branch is allowed and collect all license identifiers."""
    tokens = TOKEN.findall(expression)
    if re.sub(r"\s+", "", expression) != "".join(tokens):
        raise ValueError("invalid SPDX expression")

    def take(expected):
        if tokens and tokens[0] == expected:
            tokens.pop(0)
            return True
        return False

    def identifier():
        if not tokens or tokens[0] in {"(", ")", "AND", "OR", "WITH"}:
            raise ValueError("expected an SPDX identifier")
        return tokens.pop(0)

    identifiers = set()

    def atom():
        if take("("):
            result = alternatives()
            if not take(")"):
                raise ValueError("missing closing parenthesis in SPDX expression")
            return result
        license_id = identifier()
        license_id = ALIASES.get(license_id, license_id)
        if take("WITH"):
            license_id += f" WITH {identifier()}"
        identifiers.add(license_id)
        return license_id in allowed

    def conjunction():
        result = atom()
        while take("AND"):
            required = atom()
            result = result and required
        return result

    def alternatives():
        result = conjunction()
        while take("OR"):
            alternative = conjunction()
            result = result or alternative
        return result

    compatible = alternatives()
    if tokens:
        raise ValueError("unexpected token in SPDX expression")
    return compatible, identifiers


def check_path(root, relative, added, ports):
    path = root / relative
    if not path.is_file() or path.is_symlink():
        return []
    suffix = path.suffix.lower()
    if (
        suffix in PAYLOAD_EXTENSIONS
        or (suffix == ".bin" and path.stat().st_size >= LARGE_BINARY)
        or path.name.lower().startswith("nullmoth-v3-kit")
    ):
        return ["firmware, macOS images and optional kit payloads are acquired separately"]
    if suffix not in SOURCE_EXTENSIONS or relative.startswith("third_party/"):
        return []

    with path.open(encoding="utf-8", errors="replace") as source:
        lines = [source.readline().rstrip("\n") for _ in range(25)]
    headers = [match[1] for line in lines if (match := HEADER.match(line))]
    if not added:
        # Existing sources can lack a header, including unchanged files
        # inherited from the explicitly supplied upstream commit.
        if any(RESTRICTED.match(token) for header in headers for token in TOKEN.findall(header)):
            return ["restricted or unclassified SPDX identifier in first-party source"]
        return []
    if len(headers) != 1:
        return ["expected one SPDX-License-Identifier in the source header"]
    allowed = PORTABLE_LICENSES
    if relative.startswith("crates/stela-cli/"):
        allowed = allowed | CLI_LICENSES
    elif relative.startswith(("crates/stela-boot/", "crates/stela-linux/")):
        allowed = allowed | BOOT_LICENSES
    try:
        compatible, identifiers = check_expression(headers[0], allowed)
    except ValueError as error:
        return [str(error)]

    errors = []
    if any(RESTRICTED.match(license_id) for license_id in identifiers):
        errors.append("restricted or unclassified SPDX identifier in first-party source")
    elif not compatible:
        errors.append(f"SPDX expression {headers[0]!r} has no permitted choice for this source path")
    for license_id in sorted(identifiers):
        if license_id.endswith(" WITH Linux-syscall-note"):
            filenames = ["GPL-2.0-only.txt", "Linux-syscall-note.txt"]
        else:
            filenames = [LICENSE_TEXT.get(license_id, f"{license_id}.txt")]
        for filename in filenames:
            if not (root / "LICENSES" / filename).is_file():
                errors.append(f"missing license text LICENSES/{filename}")
    if headers[0] != "GPL-2.0-or-later" or any(PORT.search(line) for line in lines):
        if not re.search(rf"(?<![\w./-]){re.escape(relative)}(?![\w./-])", ports):
            errors.append("record the port's source project, revision and file in NOTICE (Ported sources)")
    return errors


def git_paths(*arguments):
    output = subprocess.check_output(["git", *arguments, "-z"])
    return {item.decode(errors="surrogateescape") for item in output.split(b"\0") if item}


def unchanged_upstream_paths(root, upstream_ref, added):
    """Find additions whose tracked and working contents match the upstream path."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{upstream_ref}^{{commit}}"],
        cwd=root,
    ).decode().strip()
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision], cwd=root)
    regular_modes = {b"100644", b"100755"}
    upstream_blobs = {}
    for record in tree.split(b"\0"):
        if record:
            info, path = record.split(b"\t", 1)
            mode, kind, oid = info.split()
            if mode in regular_modes and kind == b"blob":
                upstream_blobs[path] = oid

    index = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    unchanged = set()
    for record in index.split(b"\0"):
        if not record:
            continue
        info, path = record.split(b"\t", 1)
        mode, oid, stage = info.split()
        if stage != b"0" or mode not in regular_modes or oid != upstream_blobs.get(path):
            continue
        relative = path.decode(errors="surrogateescape")
        if relative not in added:
            continue
        working = root / relative
        if not working.is_file() or working.is_symlink():
            continue
        working_oid = subprocess.check_output(
            ["git", "hash-object", "--no-filters", "--", relative], cwd=root
        ).strip()
        if working_oid == oid:
            unchanged.add(relative)
    return unchanged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="base commit to compare with the checked-out source")
    parser.add_argument("--include-untracked", action="store_true", help="include local untracked files")
    parser.add_argument("--upstream-ref", help="trusted upstream commit or ref to compare inherited paths")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    added = git_paths("diff", "--name-only", "--no-renames", "--diff-filter=A", args.base)
    changed = git_paths("diff", "--name-only", "--no-renames", "--diff-filter=MT", args.base)
    if args.include_untracked:
        added |= git_paths("ls-files", "--others", "--exclude-standard")
    _, _, ports = (root / "NOTICE").read_text(encoding="utf-8").partition("Ported sources")
    inherited = unchanged_upstream_paths(root, args.upstream_ref, added) if args.upstream_ref else set()
    failed = False
    for relative in sorted(added | changed):
        for error in check_path(root, relative, relative in added and relative not in inherited, ports):
            print(f"{relative!r}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
