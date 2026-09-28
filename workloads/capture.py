#!/usr/bin/env python3
"""Standardized model -> MLIR capture driver (stable + scalable to any model).

Every model lives in ``workloads/<model>/`` with:
  - ``loader.py``    : ``get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])``  [REQUIRED]
  - ``capture.toml`` : venv / deps / env / upstream config                            [OPTIONAL]

Because models need conflicting torch/transformers/torchao versions, each gets a dedicated
venv (default ``workloads/<model>/.venv``). This driver:
  1. ensures that venv exists (creates it + installs deps from capture.toml, idempotent);
  2. runs the capture INSIDE that venv (subprocess) for each requested datatype/level;
  3. writes a single MLIR or a loader-authored multi-program session bundle, and checks 0 opaque ops;
  4. prints a summary.

Usage:
  python workloads/capture.py <model> [--formats fp32,int8,fp8] [--level linalg-on-tensors|high-level]
  python workloads/capture.py --all
  python workloads/capture.py --list

The same file is reused as the in-venv worker (``--_inner``); you never call that directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # model2MLIR/
WORKLOADS = REPO / "workloads"
BASE_DEPS = ["xdsl", "structlog", "ml_dtypes", "torchao"]  # always needed by m2m
SCHEME = {  # datatype/format -> torchao scheme (None = no torchao quantization)
    "fp32": None,
    "bf16": None,   # base weights cast to the dtype (no torchao) — see DTYPE_CAST
    "fp16": None,
    "int8": "int8_weight_only",
    # ⚠️ `int8` IS WEIGHT-ONLY, AND THAT IS A DIFFERENT PROGRAM FROM AN INTEGER DATAPATH.
    # `int8_weight_only` dequantizes the weight and emits a FLOAT matmul over it: measured on
    # tiny_llama, 155 `linalg.matmul` at `prov.orig_dtype = "float32"` over f32 tensors and zero
    # integer contractions. That is the right capture for a model whose weights are stored int8, and
    # the wrong one for measuring a systolic integer mesh -- and it cannot be repaired by substituting
    # a golden, because the program contains no integer contraction to grade.
    #
    # `int8_w8a8` quantizes the ACTIVATIONS too, so the export contains `aten._int_mm` accumulating in
    # i32 -- the arithmetic the hardware performs -- and torch eager then computes the same quantized
    # math, which makes the golden right by construction. Added as a separate FORMAT rather than by
    # changing what `int8` means: every existing int8 bundle, tolerance table and comparison is keyed
    # on the weight-only meaning, and silently redefining it would move all of them at once.
    "int8_w8a8": "int8_dyn_act_int8_weight",
    "fp8": "float8_weight_only_e4m3",
}

# Formats that are a plain dtype cast of the base weights (no torchao scheme): the model is cast to
# this torch dtype before export, so the emitted MLIR is bf16/fp16 end to end.
DTYPE_CAST = {"bf16": "bfloat16", "fp16": "float16"}


def _load_toml(model_dir: Path) -> dict:
    f = model_dir / "capture.toml"
    if not f.exists():
        return {}
    try:
        import tomllib
        return tomllib.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _venv_python(model: str, cfg: dict) -> Path:
    """Resolve the model's venv python. Default: workloads/<model>/.venv (the standard)."""
    venv = cfg.get("venv", ".venv")
    p = Path(venv)
    if not p.is_absolute():
        p = (WORKLOADS / model / p).resolve()
    return p / "bin" / "python"


def _ensure_venv(model: str, cfg: dict, *, build: bool) -> Path:
    py = _venv_python(model, cfg)
    if py.exists():
        return py
    if not build:
        raise SystemExit(f"[{model}] venv missing at {py.parent.parent} (run without --no-venv to build)")
    venv_dir = py.parent.parent
    pyver = cfg.get("python", "3.12")
    print(f"[{model}] creating venv {venv_dir} (python {pyver})", flush=True)
    subprocess.run(["uv", "venv", str(venv_dir), "--python", pyver], check=True)
    deps = list(cfg.get("deps", [])) + BASE_DEPS
    if deps:
        subprocess.run(["uv", "pip", "install", "--python", str(py), *deps], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(py), "-e", str(REPO), "--no-deps"], check=True)
    return py


def _quant_for(cfg: dict, fmt: str):
    """QuantizationConfig for a format. A model may override the default scheme via
    capture.toml ``[quant.<fmt>]`` (scheme + optional per_module map) -- e.g. BitVLA, whose
    BitNet layers can't be torchao-quantized, quantizes only its lm_head."""
    from m2m.capture.torchao_pipeline import QuantizationConfig

    override = (cfg.get("quant") or {}).get(fmt)
    if override:
        return QuantizationConfig(scheme=override.get("scheme", "none"),
                                  per_module=override.get("per_module") or None)
    scheme = SCHEME[fmt]
    return QuantizationConfig(scheme=scheme) if scheme else None


def _capture_multi_program(mdl, model_dir: Path, stem: str, quant) -> dict:
    """Use a loader's generic deferred-session writer and check its stage artifacts."""
    import hashlib
    import yaml
    from m2m.coverage import family_histogram, opaque_report

    out = model_dir / f"{stem}_session"
    if out.is_symlink() or (out.exists() and any(path.is_symlink() for path in out.rglob("*"))):
        raise ValueError("multi-program output already contains a symlink")
    summary = mdl.write_bundle(out, quant=quant)
    if not isinstance(summary, dict) or summary.get("out") != str(out):
        raise ValueError("multi-program writer did not return its exact output directory")
    contract_path = out / "session_contract.yaml"
    if out.is_symlink() or contract_path.is_symlink():
        raise ValueError("multi-program output or session contract is a symlink")
    if not contract_path.is_file():
        raise ValueError("multi-program writer omitted session_contract.yaml")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    programs = contract.get("programs", ()) if isinstance(contract, dict) else ()
    names = [str(row.get("name")) for row in programs if isinstance(row, dict)]
    if (not names or len(names) != len(programs) or len(set(names)) != len(names)
            or names != [str(name) for name in contract.get("stages", ())]
            or len(names) != summary.get("n_programs")):
        raise ValueError("multi-program writer returned an inconsistent stage roster")

    stages = []
    families = set()
    for row in programs:
        relative = Path(str(row.get("bundle", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"multi-program stage {row['name']} has an unsafe bundle path")
        for index in range(1, len(relative.parts) + 1):
            if (out / Path(*relative.parts[:index])).is_symlink():
                raise ValueError(f"multi-program stage {row['name']} bundle path contains a symlink")
        stage = out / relative
        receipt_path = stage / "capture_receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError(f"multi-program stage {row['name']} omitted a plain capture receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        artifacts = receipt.get("artifacts") if isinstance(receipt, dict) else None
        required = {"model.mlir", "weights.safetensors",
                    "weights.safetensors.manifest.json", "inputs.npz",
                    "golden.npy", "extra.npz", "input_order.json"}
        if (not isinstance(receipt, dict)
                or receipt.get("schema") != "m2m.capture-receipt.v1"
                or not isinstance(artifacts, dict) or not required.issubset(artifacts)):
            raise ValueError(f"multi-program stage {row['name']} has an incomplete capture receipt")
        for name, record in artifacts.items():
            if (not isinstance(name, str) or name in {".", ".."}
                    or Path(name).parts != (name,)):
                raise ValueError(f"multi-program stage {row['name']} has an unsafe receipt artifact name")
            artifact = stage / name
            if artifact.is_symlink() or not artifact.is_file():
                raise ValueError(f"multi-program stage {row['name']} omitted plain artifact {name}")
            if not isinstance(record, dict) or type(record.get("bytes")) is not int:
                raise ValueError(f"multi-program stage {row['name']} has malformed receipt for {name}")
            digest = hashlib.sha256()
            with artifact.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if (record["bytes"] != artifact.stat().st_size
                    or record.get("sha256") != digest.hexdigest()):
                raise ValueError(f"multi-program stage {row['name']} receipt does not bind {name}")
        mlir_bytes = (stage / "model.mlir").read_bytes()
        if not mlir_bytes:
            raise ValueError(f"multi-program stage {row['name']} has empty MLIR")
        mlir = mlir_bytes.decode("utf-8")
        opaque = opaque_report(mlir)
        stage_families = set(family_histogram(mlir))
        families.update(stage_families)
        stages.append({
            "name": row["name"], "path": str(stage / "model.mlir"),
            "receipt": str(stage / "capture_receipt.json"),
            "opaque": sum(opaque.values()), "linalg": mlir.count("linalg."),
            "families": len(stage_families), "bytes": len(mlir_bytes),
        })

    return {
        "ok": all(row["opaque"] == 0 for row in stages),
        "opaque": sum(row["opaque"] for row in stages),
        "linalg": sum(row["linalg"] for row in stages),
        "families": len(families),
        "sections": 0, "path": str(out), "bytes": sum(row["bytes"] for row in stages),
        "session_contract": str(contract_path), "n_programs": len(stages),
        "programs": stages,
        "qualification": "receipt-listed capture artifacts byte-verified; no source-closure or execution proof",
        "source_closure_verified": False,
    }


def _inner_capture(model: str, formats: list[str], level: str) -> None:
    """Runs inside the model's venv: capture each format and emit a JSON summary."""
    import m2m
    from m2m.coverage import family_histogram, opaque_report

    model_dir = WORKLOADS / model
    cfg = _load_toml(model_dir)
    sys.path.insert(0, str(model_dir))
    from loader import get_model_and_inputs  # type: ignore

    suffix = {"fp32": "", "bf16": "_bf16", "fp16": "_fp16", "int8": "_int8",
              "int8_w8a8": "_int8_w8a8", "fp8": "_fp8"}
    results = {}
    for fmt in formats:
        try:
            # re-build the model per format: torchao quantize_ mutates in place, so reusing
            # one instance across formats would double-quantize.
            mdl, inputs = get_model_and_inputs()
            q = _quant_for(cfg, fmt)
            if callable(getattr(mdl, "write_bundle", None)):
                if fmt in DTYPE_CAST:
                    raise ValueError(
                        f"multi-program capture does not support the {fmt} dtype-cast format")
                if level != "linalg-on-tensors":
                    raise ValueError("multi-program capture only supports linalg-on-tensors")
                if os.environ.get("M2M_EMIT_SECTIONS"):
                    raise ValueError("multi-program capture does not support --sections")
                results[fmt] = _capture_multi_program(
                    mdl, model_dir, f"{model}{suffix[fmt]}", q)
                continue
            if fmt in DTYPE_CAST:  # base-weight dtype cast (bf16/fp16); no torchao
                import torch
                mdl = mdl.to(getattr(torch, DTYPE_CAST[fmt]))
            weights_path = str(model_dir / f"{model}{suffix[fmt]}.safetensors")
            r = m2m.convert(mdl, inputs, backend="fx_importer", quantization=q, level=level,
                            weights_path=weights_path)
            path = model_dir / f"{model}{suffix[fmt]}.mlir"
            path.write_text(r.mlir_text)
            opaque = opaque_report(r.mlir_text)
            n_sections = 0
            if os.environ.get("M2M_EMIT_SECTIONS") and r.module is not None:
                try:
                    from m2m.api import module_to_text
                    from m2m.transforms import split_by_section
                    secdir = model_dir / "sections"
                    secdir.mkdir(exist_ok=True)
                    secs = split_by_section(r.module)
                    for sname, smod in secs.items():
                        (secdir / f"{model}{suffix[fmt]}.{sname}.mlir").write_text(module_to_text(smod))
                    n_sections = len(secs)
                except Exception:  # noqa: BLE001
                    pass
            results[fmt] = {
                "ok": r.ok, "opaque": sum(opaque.values()), "opaque_detail": opaque,
                "linalg": r.mlir_text.count("linalg."),
                "families": len(family_histogram(r.mlir_text)),
                "sections": n_sections,
                "path": str(path), "bytes": len(r.mlir_text),
            }
        except Exception as exc:  # noqa: BLE001 - one format failing must not sink the others
            import traceback
            results[fmt] = {"ok": False, "opaque": -1, "error": str(exc)[:400],
                            "trace": traceback.format_exc()[-600:]}
    print("__CAPTURE_RESULT__ " + json.dumps(results))


def _run_one(model: str, formats: list[str], level: str, *, build: bool) -> dict:
    cfg = _load_toml(WORKLOADS / model)
    py = _ensure_venv(model, cfg, build=build)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    cmd = [str(py), str(Path(__file__).resolve()), "--_inner", model,
           "--formats", ",".join(formats), "--level", level]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("__CAPTURE_RESULT__ "):
            return json.loads(line[len("__CAPTURE_RESULT__ "):])
    raise RuntimeError(f"[{model}] capture failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def _all_models() -> list[str]:
    return sorted(d.name for d in WORKLOADS.iterdir()
                  if d.is_dir() and (d / "loader.py").exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="model name (a workloads/<model>/ dir)")
    ap.add_argument("--formats", default="fp32,int8,fp8", help="comma list of: fp32,int8,fp8")
    ap.add_argument("--level", default="linalg-on-tensors", choices=["linalg-on-tensors", "high-level"])
    ap.add_argument("--all", action="store_true", help="capture every model with a loader.py")
    ap.add_argument("--list", action="store_true", help="list available models")
    ap.add_argument("--no-venv", action="store_true", help="don't build missing venvs (fail instead)")
    ap.add_argument("--sections", action="store_true", help="also emit per-source-module section .mlir files")
    ap.add_argument("--_inner", dest="inner", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.list:
        print("\n".join(_all_models()))
        return 0

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in SCHEME]
    if bad:
        raise SystemExit(f"unknown formats: {bad} (valid: {list(SCHEME)})")

    if args.inner:  # in-venv worker
        _inner_capture(args.model, formats, args.level)
        return 0

    if args.sections:
        os.environ["M2M_EMIT_SECTIONS"] = "1"  # propagated to the in-venv worker

    models = _all_models() if args.all else ([args.model] if args.model else [])
    if not models:
        ap.error("give a model name or --all (or --list)")

    failures = 0
    for model in models:
        try:
            res = _run_one(model, formats, args.level, build=not args.no_venv)
            for fmt, r in res.items():
                clean = r["ok"] and r["opaque"] == 0
                if not clean:
                    failures += 1
                if "error" in r:
                    print(f"!! {model:14s} {fmt:5s} ERROR: {r['error']}")
                else:
                    print(f"{'OK ' if clean else '!! '}{model:14s} {fmt:5s} "
                          f"opaque={r['opaque']:<4d} linalg={r['linalg']:<5d} "
                          f"families={r['families']:<3d} -> {r['path']}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"!! {model:14s} FAILED: {str(exc)[:300]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
