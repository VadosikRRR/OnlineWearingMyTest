"""No-install verification: Python source, notebook cells, imports and unit tests.

Run with Python 3.11+ from any working directory. No ML packages or weights needed.
"""
import ast
import compileall
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required")
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert configuration["project"]["requires-python"] == ">=3.11"
    vendor_manifest = json.loads((ROOT / "docs/vendor_manifest.json").read_text())
    for item in vendor_manifest["files"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
    files = sorted(path for folder in ("src", "scripts", "tests") for path in (ROOT / folder).rglob("*.py"))
    for path in files:
        source = path.read_text(encoding="utf-8")
        # Parse with the 3.11 grammar even when the local interpreter is 3.12+.
        ast.parse(source, filename=str(path), feature_version=(3, 11))
        compile(source, str(path), "exec")
    assert compileall.compile_dir(ROOT / "src", quiet=1)
    notebook = json.loads((ROOT / "notebooks/kaggle_kgi_end_to_end.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kaggle"]["isGpuEnabled"] is False
    code_cells = 0
    for number, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in ("code", "markdown")
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source, filename=f"notebook cell {number}", feature_version=(3, 11))
            compile(source, f"notebook cell {number}", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            code_cells += 1
    sys.path.insert(0, str(ROOT / "src"))
    import online_wearing
    assert "torch" not in sys.modules
    assert "numpy" not in sys.modules
    assert online_wearing.Weights is not None
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
                   cwd=ROOT, check=True)
    print(f"PASS: {len(files)} Python files, Python 3.11 grammar, {code_cells} notebook code cells, safe import, unit tests")


if __name__ == "__main__":
    main()
