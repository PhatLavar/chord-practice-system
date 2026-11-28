"""Check package/notebook/docs paths; optionally execute real offline notebook smoke cells."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check_static():
    notebooks = []
    commands = set()
    links = 0
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(doc["cells"]):
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            tree = ast.parse(text, filename=f"{path}:{i}")
            compile(tree, str(path), "exec")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.List)
                    and node.elts
                    and ast.unparse(node.elts[0]) == "sys.executable"
                ):
                    for item in node.elts[1:]:
                        if isinstance(item, ast.Constant) and isinstance(item.value, str):
                            if item.value.startswith("src/acr/"):
                                raise ValueError(f"Legacy script CLI in {path}")
                            if item.value.startswith("acr."):
                                if importlib.util.find_spec(item.value) is None:
                                    raise ValueError(f"Missing module {item.value}")
                                commands.add(item.value)
        notebooks.append(path.relative_to(ROOT).as_posix())
    for path in [
        ROOT / "README.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "notebooks").rglob("*.md"),
    ]:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https:", "http:", "mailto:", "#")):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                raise ValueError(f"Broken documentation link: {path}: {target}")
            links += 1

    def config_refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_config") and isinstance(item, str) and item.endswith(".json"):
                    if not (ROOT / item).is_file():
                        raise ValueError("Missing config " + item)
                config_refs(item)
        elif isinstance(value, list):
            for item in value:
                config_refs(item)

    configs = list((ROOT / "configs").rglob("*.json"))
    for path in configs:
        config_refs(json.loads(path.read_text(encoding="utf-8")))
    return dict(
        notebooks=notebooks,
        code_cells_compile=True,
        notebook_cli_modules=sorted(commands),
        local_documentation_links=links,
        configs_checked=len(configs),
    )


def run_smoke(output):
    import hashlib
    import os
    import subprocess
    import sys
    import torch

    if output.exists():
        raise FileExistsError("Use a fresh smoke output directory")
    for name in ("annotation.zip", "audio_mono-mic.zip"):
        if not (ROOT / "data/raw/guitarset" / name).is_file():
            raise FileNotFoundError("Smoke checks require previously downloaded GuitarSet archives")
    output.mkdir(parents=True)
    original_cwd = Path.cwd()
    results = []
    try:
        os.chdir(ROOT)
        # Execute the actual P0 code without IPython rendering, into a fresh output.
        path = ROOT / "notebooks/p0/00_data_audit.ipynb"
        cell = next(c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code")

        class P0Local(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == "IPython.display" else node

            def visit_Expr(self, node):
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "display"
                ):
                    return None
                return self.generic_visit(node)

            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id == "output" for t in node.targets):
                    node.value = ast.Call(
                        func=ast.Name(id="Path", ctx=ast.Load()),
                        args=[ast.Constant(str(output / "p0"))],
                        keywords=[],
                    )
                return node

        tree = ast.fix_missing_locations(P0Local().visit(ast.parse("".join(cell["source"]))))
        os.chdir(path.parent)
        exec(compile(tree, str(path), "exec"), {})
        os.chdir(ROOT)
        results.append(
            dict(
                notebook=path.relative_to(ROOT).as_posix(),
                cells="audit, without IPython rendering",
                passed=True,
            )
        )

        class LocalOutputs(ast.NodeTransformer):
            def __init__(self, folder):
                self.folder = folder

            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value.startswith("runs/"):
                    return ast.copy_location(
                        ast.Constant(str(self.folder / Path(node.value).name)), node
                    )
                return node

        for prefix in ("02h", "02i", "02j", "02k"):
            path = next((ROOT / "notebooks/p2/guitarset").glob(prefix + "*.ipynb"))
            doc = json.loads(path.read_text(encoding="utf-8"))
            scope = dict(
                Path=Path,
                subprocess=subprocess,
                sys=sys,
                os=os,
                json=json,
                hashlib=hashlib,
                torch=torch,
                project=ROOT,
            )
            # Cells 2/3 are unchanged input/cache checks and real CPU smoke launches.
            # Clone, pip install, GPU training, plots and ZIP export are intentionally excluded.
            for index in (2, 3):
                source = "".join(doc["cells"][index]["source"])
                tree = ast.fix_missing_locations(
                    LocalOutputs(output / path.stem).visit(ast.parse(source))
                )
                exec(compile(tree, f"{path}:cell{index}", "exec"), scope)
            results.append(
                dict(notebook=path.relative_to(ROOT).as_posix(), cells=[2, 3], passed=True)
            )
    finally:
        os.chdir(original_cwd)
    return dict(
        checks=results,
        scope="Real local code-cell execution with installed dependencies and existing archives; fresh output paths. No remote clone/install, GUI rendering or full Kaggle GPU replay.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "runs/refactor-validation/notebook-smoke")
    args = p.parse_args()
    report = check_static()
    if args.smoke:
        report["execution"] = run_smoke(args.output.resolve())
    target = ROOT / "runs/refactor-validation/notebook-checks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
