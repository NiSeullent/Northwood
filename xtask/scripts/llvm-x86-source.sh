#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

if [ "$#" -ne 1 ]; then
    printf 'Run %s NEW_SOURCE_DIRECTORY\n' "$0" >&2
    exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
pin=$(git -C "$root" ls-files --stage -- third_party/llvm |
    awk '$1 == "160000" && $3 == "0" {print $2}')
case "$pin" in
    ''|*[!0-9a-f]*)
        printf '%s\n' 'The repository index must pin third_party/llvm.' >&2
        exit 1
        ;;
esac
url=$(git config -f "$root/.gitmodules" --get submodule.llvm.url)

mkdir -- "$1"
checkout=$(CDPATH= cd -- "$1" && pwd -P)
git -C "$checkout" init -q
git -C "$checkout" config core.autocrlf false
git -C "$checkout" remote add origin "$url"
{
    cat "$root/third_party/llvm.sparse"
    # CMake still generates common target-parser tables for these targets.
    printf '%s\n' \
        '/llvm/lib/Target/X86/' \
        '/llvm/lib/Target/ARM/**/*.td' \
        '/llvm/lib/Target/AArch64/**/*.td' \
        '/llvm/lib/Target/RISCV/**/*.td' \
        '/llvm/lib/Target/PowerPC/**/*.td' \
        '/llvm/projects/CMakeLists.txt'
} | git -C "$checkout" sparse-checkout set --no-cone --stdin
git -C "$checkout" fetch -q --no-tags --depth=1 --filter=blob:none origin "$pin"
git -C "$checkout" checkout -q --detach "$pin"
printf '%s\n' "$checkout"
