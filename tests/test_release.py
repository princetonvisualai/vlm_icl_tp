import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

FORBIDDEN = ("vlm_inference", "/scratch/", "/home/", "bw5889", "OLGARUS")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def shipped_files():
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    for rel in filter(None, out.split("\0")):
        path = ROOT / rel
        if path.is_file() and path.resolve() != SELF:
            yield rel, path


def test_no_private_paths_or_emails():
    hits = []
    for rel, path in shipped_files():
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        for lineno, line in enumerate(raw.decode("utf-8", errors="ignore").splitlines(), 1):
            if any(marker in line for marker in FORBIDDEN) or EMAIL.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:120]}")
    assert not hits, "private paths or emails in shipped files:\n" + "\n".join(hits)


def test_license_and_readme_exist():
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    assert (ROOT / "README.md").is_file()
