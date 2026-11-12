# SPDX-License-Identifier: GPL-2.0-or-later

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_licenses import LARGE_BINARY, check_expression, check_path


CHECKER = Path(__file__).resolve().with_name("check_licenses.py")


class LicenseCheckTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for license_id in ("GPL-2.0-only", "BSD-3-Clause", "Apache-2.0", "MIT"):
            self.write(f"LICENSES/{license_id}.txt", b"license fixture")

    def write(self, relative, contents):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
             *args], cwd=self.root, text=True
        ).strip()

    def commit(self, title):
        self.git("add", ".")
        self.git("commit", "--quiet", "--no-gpg-sign", "-m", title)
        return self.git("rev-parse", "HEAD")

    def upstream_history(self, files):
        self.git("init", "--quiet")
        self.write("NOTICE", b"Ported sources\n")
        base = self.commit("Base")
        for path, contents in files.items():
            self.write(path, contents)
        upstream = self.commit("Upstream")
        self.git("checkout", "--quiet", "--detach", base)
        return base, upstream

    def run_checker(self, base, *options):
        return subprocess.run(
            [sys.executable, "-B", str(CHECKER), base, *options],
            cwd=self.root, capture_output=True, text=True
        )

    def check_source(self, relative, expression, ports="", added=True):
        text = f"// SPDX-License-Identifier: {expression}\n" if expression else "fn main() {}\n"
        self.write(relative, text.encode())
        return check_path(self.root, relative, added, ports)

    def test_new_file_needs_header_but_existing_upstream_file_does_not(self):
        path = "crates/stela-boot/src/example.rs"
        self.assertEqual(len(self.check_source(path, "")), 1)
        self.assertEqual(self.check_source(path, "", added=False), [])
        self.assertEqual(self.check_source(path, "GPL-2.0-or-later"), [])

    def test_exact_tracked_upstream_source_can_lack_header(self):
        path = "src/upstream.rs"
        contents = b"fn main() {}\n"
        base, upstream = self.upstream_history({path: contents})
        self.write(path, contents)
        self.commit("Import from upstream")

        self.assertEqual(self.run_checker(base).returncode, 1)
        result = self.run_checker(base, "--upstream-ref", upstream)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_missing_and_untracked_upstream_paths_need_headers(self):
        contents = b"fn main() {}\n"
        upstream_files = dict.fromkeys(
            ("src/changed.rs", "src/local_edit.rs", "src/untracked.rs"), contents
        )
        base, upstream = self.upstream_history(upstream_files)
        self.write("src/changed.rs", contents + b"// committed edit\n")
        self.write("src/local_edit.rs", contents)
        self.write("src/new_path.rs", contents)
        self.commit("Add sources")
        self.write("src/local_edit.rs", contents + b"// working tree edit\n")
        self.write("src/untracked.rs", contents)

        result = self.run_checker(base, "--upstream-ref", upstream, "--include-untracked")
        self.assertEqual(result.returncode, 1)
        for path in (*upstream_files, "src/new_path.rs"):
            self.assertIn(f"{path!r}: expected one SPDX-License-Identifier", result.stderr)

    def test_identical_upstream_source_and_payload_still_face_restrictions(self):
        source = "src/restricted.rs"
        payload = "assets/firmware.ucode"
        files = {
            source: b"// SPDX-License-Identifier: MIT OR PolyForm-Noncommercial-1.0.0\n",
            payload: b"firmware fixture",
        }
        base, upstream = self.upstream_history(files)
        for path, contents in files.items():
            self.write(path, contents)
        self.commit("Import from upstream")

        result = self.run_checker(base, "--upstream-ref", upstream)
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"{source!r}: restricted or unclassified SPDX identifier", result.stderr)
        self.assertIn(f"{payload!r}: firmware, macOS images", result.stderr)

    def test_boot_and_cli_license_choices(self):
        cli = "crates/stela-cli/src/example.rs"
        boot = "crates/stela-boot/src/example.rs"
        self.assertEqual(self.check_source(boot, "GPL-2.0-only", boot), [])
        self.assertEqual(self.check_source(cli, "Apache-2.0", cli), [])
        self.assertIn("no permitted choice", self.check_source(cli, "GPL-2.0-only", cli)[0])
        self.assertIn("no permitted choice", self.check_source(boot, "Apache-2.0", boot)[0])
        self.assertEqual(self.check_source(cli, "GPL-2.0 OR BSD-3-Clause", cli), [])
        self.assertIn("no permitted choice", self.check_source(cli, "GPL-2.0 AND BSD-3-Clause", cli)[0])

    def test_spdx_parentheses_precedence_and_all_branches(self):
        self.assertEqual(check_expression("MIT OR BSD-3-Clause AND GPL-2.0", {"MIT"}),
                         (True, {"MIT", "BSD-3-Clause", "GPL-2.0-only"}))
        self.assertFalse(check_expression("(MIT OR BSD-3-Clause) AND GPL-2.0", {"MIT"})[0])
        with self.assertRaises(ValueError):
            check_expression("MIT OR (BSD-3-Clause", {"MIT"})

    def test_restricted_alternative_is_rejected(self):
        path = "crates/stela-boot/src/example.rs"
        for added in (True, False):
            errors = self.check_source(path, "MIT OR PolyForm-Noncommercial-1.0.0", path, added)
            self.assertTrue(any("restricted" in error for error in errors))

    def test_port_requires_license_text_and_an_exact_notice_path(self):
        path = "crates/stela-boot/src/example.rs"
        errors = self.check_source(path, "MIT", f"{path}.backup")
        self.assertTrue(any("NOTICE" in error for error in errors))
        (self.root / "LICENSES/MIT.txt").unlink()
        self.assertEqual(self.check_source(path, "MIT", f"{path}: a port"),
                         ["missing license text LICENSES/MIT.txt"])

    def test_firmware_and_large_binaries_are_rejected(self):
        for path, size in (("assets/fw.ucode", 1), ("assets/apple.ipsw", 1),
                           ("assets/payload.bin", LARGE_BINARY)):
            self.write(path, b"\0" * size)
            self.assertEqual(len(check_path(self.root, path, True, "")), 1)
        self.write("assets/font.bin", b"\0")
        self.assertEqual(check_path(self.root, "assets/font.bin", True, ""), [])


if __name__ == "__main__":
    unittest.main()
