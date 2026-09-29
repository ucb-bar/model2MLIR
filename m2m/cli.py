"""m2m command-line interface.

    m2m convert  <model.py> [--out out.mlir] [--output-type ...] [--quant SCHEME]
    m2m coverage <model.py>

`<model.py>` must expose `get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])`
(see examples/simple_mlp.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_model_module(path: str) -> Any:
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model module from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_model_and_inputs"):
        raise RuntimeError(f"{p} must define get_model_and_inputs() -> (model, inputs)")
    return module.get_model_and_inputs()


def _make_quant(scheme: str | None) -> Any:
    if not scheme:
        return None
    from m2m.capture.torchao_pipeline import QuantizationConfig

    return QuantizationConfig(scheme=scheme)


def _cmd_convert(args: argparse.Namespace) -> int:
    import m2m

    model, inputs = _load_model_module(args.model)
    quantization = _make_quant(args.quant)
    if args.quant_adapter:
        from m2m.capture.external_quantization import ExternalQuantizationConfig

        if args.quant or not args.contract or not args.policy:
            raise ValueError("--quant-adapter requires --contract and --policy, without --quant")
        quantization = ExternalQuantizationConfig(
            args.quant_adapter, args.contract, args.policy,
            args.contract_sha256, args.policy_sha256,
        )
    result = m2m.convert(
        model,
        tuple(inputs),
        output_type=args.output_type,
        quantization=quantization,
        backend=getattr(args, "backend", "auto"),
        level=getattr(args, "level", "linalg-on-tensors"),
        capture_trace=bool(args.quant_adapter),
    )
    if not result.ok:
        sys.stderr.write("conversion failed:\n  " + "\n  ".join(result.diagnostics) + "\n")
        return 1
    if args.out:
        Path(args.out).write_text(result.mlir_text)
        if result.quantization_manifest is not None:
            import json

            Path(args.out + ".quantization.json").write_text(
                json.dumps(result.quantization_manifest, sort_keys=True, indent=2) + "\n"
            )
        sys.stderr.write(
            f"[{result.frontend}/{result.path_taken}] {result.output_type}: "
            f"wrote {len(result.mlir_text)} chars to {args.out}\n"
        )
    else:
        sys.stdout.write(result.mlir_text)
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    import json

    import m2m

    model, inputs = _load_model_module(args.model)
    report = m2m.coverage_report(model, tuple(inputs), quantization=_make_quant(args.quant))
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="m2m", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # `convert` auto-detects torch vs jax; `lower-torch` / `lower-jax` are explicit
    # aliases (same handler) matching the /m2m-lower-* skills.
    for cmd, helptext in (
        ("convert", "convert a model to MLIR (auto-detect torch/jax)"),
        ("lower-torch", "lower a PyTorch model (loader: get_model_and_inputs) to MLIR"),
        ("lower-jax", "lower a JAX function (loader: get_model_and_inputs -> (fn, inputs)) to StableHLO"),
    ):
        pc = sub.add_parser(cmd, help=helptext)
        pc.add_argument("model", help="path to a .py exposing get_model_and_inputs()")
        pc.add_argument("--out", help="output .mlir path (default: stdout)")
        pc.add_argument("--output-type", default="linalg-on-tensors")
        pc.add_argument("--backend", default="auto", choices=["auto", "torch_mlir", "fx_importer"])
        pc.add_argument("--quant", default=None, help="torchAO scheme name (e.g. int8_weight_only)")
        pc.add_argument("--quant-adapter", default=None, help="installed external quantization adapter ID")
        pc.add_argument("--contract", default=None, help="selected external software contract")
        pc.add_argument("--policy", default=None, help="selected external model quantization policy")
        pc.add_argument("--contract-sha256", default=None, help="expected SHA-256 of selected contract")
        pc.add_argument("--policy-sha256", default=None, help="expected SHA-256 of selected policy")
        pc.add_argument("--level", default="linalg-on-tensors",
                        choices=["linalg-on-tensors", "high-level"], help="representation level")
        pc.set_defaults(func=_cmd_convert)

    pv = sub.add_parser("coverage", help="report op coverage for a model")
    pv.add_argument("model", help="path to a .py exposing get_model_and_inputs()")
    pv.add_argument("--quant", default=None)
    pv.set_defaults(func=_cmd_coverage)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
