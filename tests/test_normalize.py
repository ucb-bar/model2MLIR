"""Provider normalization runs before the final text and trace are sealed."""
from hashlib import sha256

import pytest
import torch
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.tensor import CastOp
from xdsl.rewriter import Rewriter

import m2m
from m2m.capture.trace import copy_mlir_sources, snapshot_exported_program
from m2m.coverage import opaque_report


_LIB = torch.library.Library("m2m_normalize_test", "DEF")
_LIB.define("identity(Tensor x) -> Tensor")
_LIB.impl("identity", lambda x: x.clone(), "CompositeExplicitAutograd")
torch.library.register_fake("m2m_normalize_test::identity")(lambda x: x.clone())


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.ops.m2m_normalize_test.identity.default(x)


@pytest.mark.parametrize("opaque_custom_ops", [False, True])
def test_identity_elimination_trace_alias_evidence_requires_opt_in(opaque_custom_ops):
    from m2m.capture.torch_export import _prepare_exported_program

    exported = torch.export.export(Model(), (torch.ones(2, 3),))
    snapshot_exported_program(exported, stage="original")
    producer = next(node for node in exported.graph.nodes if node.op == "call_function")
    input_ids = list(producer.args[0].meta["custom"]["m2m_lineage"])
    expected = {"source_ids": list(producer.meta["custom"]["m2m_lineage"]),
                "kind": "eliminated", "reason": "identity decomposition", "argument_index": 0}
    prepared, _ = _prepare_exported_program(
        exported, run_default_decompositions=False,
        export_decomposition_table={torch.ops.m2m_normalize_test.identity.default: lambda x: x},
        capture_trace=True, opaque_custom_ops=opaque_custom_ops)
    graph = snapshot_exported_program(prepared, stage="prepared")
    row, = graph["eliminations"]
    if opaque_custom_ops:
        from m2m.capture.trace import _literal

        expected.update(alias_input_source_ids=input_ids, alias_input_result_index=0,
                        alias_input_abi=_literal(torch.ones(2, 3)))
    assert row == expected


@pytest.mark.parametrize("normalize_enabled", [False, True])
def test_convert_noop_conversion_aliases_require_normalization(normalize_enabled):
    class NoopCast(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.to.dtype(x, torch.float32) + 1

    result = m2m.convert(
        NoopCast(), (torch.ones(2, 3),), backend="fx_importer", capture_trace=True,
        normalize=(lambda module, manifest: module) if normalize_enabled else None)
    assert result.ok and result.capture_trace["status"] == "complete", result.diagnostics
    assert result.capture_trace["graphs"]["prepared"]["eliminations"] == []
    main, = [op for op in result.module.body.block.ops
             if isinstance(op, FuncOp) and op.body.blocks]
    if normalize_enabled:
        aliases = main.arg_attrs.data[0].data["prov.value_aliases"].data
        cast = next(node for node in result.capture_trace["graphs"]["quantized"]["nodes"]
                    if node["target"] == "aten.to.dtype")
        assert cast["id"] in aliases
    else:
        assert main.arg_attrs is None


def test_normalization_precedes_final_text_trace_and_opaque_count():
    calls = []

    def normalize(module, manifest):
        calls.append((module, manifest))
        module = module.clone()
        call, = [op for op in module.walk() if isinstance(op, CallOp)]
        replacement = CastOp(call.arguments[0], call.res[0].type)
        copy_mlir_sources(replacement, call, role="lowering")
        Rewriter.replace_op(call, replacement)
        for op in list(module.body.block.ops):
            if isinstance(op, FuncOp) and not op.body.blocks:
                module.body.block.erase_op(op)
        module.attributes["test.normalized"] = StringAttr("yes")
        return module

    result = m2m.convert(Model(), (torch.ones(2, 3),), backend="fx_importer",
                         decompose=False, capture_trace=True, normalize=normalize)
    assert len(calls) == 1 and calls[0][1] is None
    assert result.ok, result.diagnostics
    assert result.module is not calls[0][0]
    assert result.module.attributes["test.normalized"].data == "yes"
    assert 'test.normalized = "yes"' in result.mlir_text
    assert opaque_report(result.mlir_text) == {}
    assert "tensor.cast" in result.mlir_text
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    assert result.capture_trace["mlir"]["sha256"] == sha256(result.mlir_text.encode()).hexdigest()
    assert not any(op["operation"] == "func.call" for op in result.capture_trace["mlir"]["operations"])


@pytest.mark.parametrize("case,error,message", [
    ("missing_trace", ValueError, "requires capture_trace=True"),
    ("nonmodule", TypeError, "must return a ModuleOp"),
    ("invalid_module", Exception, "callee"),
    ("exception", ValueError, "refused unmatched"),
])
def test_normalize_refuses_invalid_capture(case, error, message):
    def normalize(module, manifest):
        if case == "nonmodule":
            return None
        if case == "invalid_module":
            call = next(op for op in module.walk() if isinstance(op, CallOp))
            call.properties["callee"] = StringAttr("invalid symbol reference")
        if case == "exception":
            raise ValueError("normalizer refused unmatched custom call")
        return module

    with pytest.raises(error, match=message):
        m2m.convert(Model(), (torch.ones(2, 3),), backend="fx_importer",
                    capture_trace=case != "missing_trace", normalize=normalize)


def test_normalize_receives_validated_external_manifest(tmp_path):
    from m2m.capture.external_quantization import ExternalQuantizationConfig

    contract, policy = tmp_path / "contract.json", tmp_path / "policy.json"
    contract.write_bytes(b"{}")
    policy.write_bytes(b"[]")
    manifest = {"schema": "m2m.quantization_manifest.v2", "adapter_id": "fixture",
                "contract_sha256": sha256(contract.read_bytes()).hexdigest(),
                "policy_sha256": sha256(policy.read_bytes()).hexdigest(),
                "sites": [{"site_id": "host", "status": "host"}]}
    exported = torch.export.export(Model(), (torch.ones(2, 3),))
    exported._m2m_external_quantization_manifest = manifest
    seen = []

    def normalize(module, selected):
        seen.append(selected)
        assert "prov.quantization_manifest_sha256" in module.attributes
        return module

    config = ExternalQuantizationConfig("fixture", contract, policy)
    result = m2m.convert(exported, (torch.ones(2, 3),), backend="fx_importer",
                         quantization=config, quantization_preapplied=True,
                         capture_trace=True, normalize=normalize)
    assert result.ok, result.diagnostics
    assert seen == [manifest] and seen[0] is manifest
    manifest["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="policy_sha256"):
        m2m.convert(exported, (torch.ones(2, 3),), backend="fx_importer",
                    quantization=config, quantization_preapplied=True,
                    capture_trace=True, normalize=normalize)
    assert len(seen) == 1
