"""Exercise CI dependency setup without installing packages or using the network."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
STEP = re.compile(
    r"^      - name: Install (?:media subprocesses \(ffmpeg, ffprobe\)"
    r"|CUDA build headers \(no GPU required\))\n"
    r"(?P<meta>(?:        (?!run:)\S[^\n]*\n)*)"
    r"        run: \|\n(?P<body>(?:          [^\n]*\n|\n)+)",
    re.MULTILINE,
)
STUB = r"""
import json
import os
from pathlib import Path
import subprocess
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with Path(os.environ["CASE_LOG"]).open("a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if name in {"ffmpeg", "ffprobe"}:
    raise SystemExit(0)
if name == "curl":
    Path(args[args.index("--output") + 1]).write_text("keyring")
elif name == "dpkg-query":
    print("cuda-crt-13-2=13.2.1")
elif name == "sudo" and args[0] == "sed":
    for path in args[3:]:
        data = subprocess.check_output(["/usr/bin/sed", "-e", args[2], path])
        Path(path).write_bytes(data)
elif name == "sudo" and args[0] == "apt-get":
    effective = Path(os.environ["CASE_EFFECTIVE"]).read_text()
    if "azure.archive" in effective or "https://archive.ubuntu.com/ubuntu" not in effective:
        raise SystemExit(91)
    if os.environ["CASE_FAILURE"] == "update" and "update" in args:
        raise SystemExit(9)
    if "install" in args:
        if os.environ["CASE_FAILURE"] == "install":
            raise SystemExit(7)
        tool = Path(os.environ["CASE_BIN"]) / "sudo"
        for binary in ["ffmpeg", "ffprobe"]:
            destination = tool.with_name(binary)
            destination.write_bytes(tool.read_bytes())
            destination.chmod(0o755)
        Path(os.environ["CASE_HEADER"]).touch()
elif name != "sudo" or args[0] != "dpkg":
    raise SystemExit(78)
"""


def setup_steps():
    source = WORKFLOW.read_text()
    found = {}
    for match in STEP.finditer(source):
        job = re.findall(r"^  ([a-z-]+):$", source[: match.start()], re.MULTILINE)[-1]
        found[job] = (match["meta"], textwrap.dedent(match["body"]))
    return found


class CiSetupTests(unittest.TestCase):
    def test_setup_deadlines_and_shell_syntax(self):
        steps = setup_steps()
        self.assertEqual(set(steps), {"rust", "cuda", "packages"})
        for job, (meta, script) in steps.items():
            with self.subTest(job=job):
                timeout = int(re.search(r"timeout-minutes: (\d+)", meta)[1])
                self.assertTrue(1 <= timeout <= 5)
                subprocess.run([shutil.which("bash"), "-n"], input=script, text=True, check=True)

    def test_setup_shortcuts_failures_and_mirror_layouts(self):
        for job, (_, script) in setup_steps().items():
            cases = ["present", "missing", "install"] if job == "rust" else ["missing", "update"]
            for layout in ["direct", "mirror-file"]:
                for case in cases:
                    with self.subTest(job=job, layout=layout, case=case):
                        self.run_case(job, script, layout, case)

    def run_case(self, job, script, layout, case):
        with tempfile.TemporaryDirectory(prefix="ci-setup-") as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            apt = root / "apt"
            (apt / "sources.list.d").mkdir(parents=True)
            sources = apt / "sources.list.d/ubuntu.sources"
            mirrors = apt / "apt-mirrors.txt"
            mirrors.write_text("http://azure.archive.ubuntu.com/ubuntu\n")
            uri = (
                "http://azure.archive.ubuntu.com/ubuntu"
                if layout == "direct"
                else "mirror+file:/etc/apt/apt-mirrors.txt"
            )
            original = f"Types: deb\nURIs: {uri}\nSuites: noble\nSigned-By: ubuntu-keyring.gpg\n"
            sources.write_text(original)
            vendor = apt / "sources.list.d/vendor.list"
            vendor.write_text("deb https://example.invalid/ubuntu noble main\n")
            header = root / "host_config.h"
            log = root / "commands.log"
            log.touch()
            github_env = root / "github-env"
            github_env.touch()
            stub = f"#!{sys.executable}\n" + STUB
            for name in ["sudo", "curl", "dpkg-query"]:
                path = binaries / name
                path.write_text(stub)
                path.chmod(0o755)
            for name in ["sort", "paste"]:
                (binaries / name).symlink_to(shutil.which(name))
            if case == "present":
                for name in ["ffmpeg", "ffprobe"]:
                    (binaries / name).write_text(stub)
                    (binaries / name).chmod(0o755)
            script = script.replace("/etc/apt/", f"{apt}/").replace(
                "/usr/local/cuda-13.2/include/crt/host_config.h", str(header)
            )
            env = dict(
                os.environ,
                PATH=str(binaries),
                CASE_BIN=str(binaries),
                CASE_HEADER=str(header),
                CASE_LOG=str(log),
                RUNNER_TEMP=str(root),
                GITHUB_ENV=str(github_env),
                CASE_EFFECTIVE=str(sources if layout == "direct" else mirrors),
                CASE_FAILURE=case,
            )
            result = subprocess.run(
                [shutil.which("bash"), "-euo", "pipefail", "-c", script],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                result.returncode,
                {"install": 7, "update": 9}.get(case, 0),
                result.stdout + result.stderr + log.read_text(),
            )
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            apt_calls = [
                args for name, *args in commands if name == "sudo" and args[0] == "apt-get"
            ]
            if case == "present":
                self.assertFalse(apt_calls)
                self.assertEqual(sources.read_text(), original)
            else:
                self.assertEqual(len(apt_calls), 1 if case == "update" else 2)
                for args in apt_calls:
                    self.assertIn("Acquire::Retries=1", args)
                    self.assertIn("Acquire::http::Timeout=20", args)
                    self.assertIn("DPkg::Lock::Timeout=60", args)
                if job != "rust" and case == "missing":
                    self.assertIn("CUDA_TOOLKIT_CACHE_KEY=", github_env.read_text())
                if case in {"install", "update"}:
                    self.assertFalse(github_env.read_text())
            self.assertEqual(vendor.read_text(), "deb https://example.invalid/ubuntu noble main\n")


if __name__ == "__main__":
    unittest.main()
