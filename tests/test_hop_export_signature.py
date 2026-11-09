"""Inlining keeps ExportedProgram value slots and its verifier consistent."""
import copy

import pytest
import torch
from m2m.ir.torchmlir_decomps import inline_set_grad_hops


class NestedOutputs(torch.nn.Module):
    def forward(self, input):
        with torch.no_grad():
            value = input[:, 1:]
            with torch.no_grad():
                result = value.sin()
            return result, value, result, 7, None


def _export():
    value = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    return torch.export.export(NestedOutputs(), (value,)), value


def test_nested_output_slots_names_aliases_and_literals():
    exported, value = _export()
    before = exported.module()(value)
    kinds = [spec.kind for spec in exported.graph_signature.output_specs]
    targets = [spec.target for spec in exported.graph_signature.output_specs]
    call_spec = copy.deepcopy(exported.call_spec)
    assert inline_set_grad_hops(exported.graph_module, graph_signature=exported.graph_signature)
    exported.validate()  # The original public verifier is mandatory.
    after = exported.module()(value)
    assert kinds == [spec.kind for spec in exported.graph_signature.output_specs]
    assert targets == [spec.target for spec in exported.graph_signature.output_specs]
    assert exported.call_spec == call_spec
    for old, new in zip(before, after, strict=True):
        if isinstance(old, torch.Tensor):
            assert torch.equal(old.view(torch.int32), new.view(torch.int32))
            assert old.shape == new.shape and old.stride() == new.stride()
        else:
            assert type(old) is type(new) and old == new
    assert after[0] is after[2]
    assert after[1].untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    assert after[1].storage_offset() == 1
    assert not inline_set_grad_hops(exported.graph_module, graph_signature=exported.graph_signature)


def test_graph_only_inlining_does_not_silently_repair_exported_signature():
    exported, _ = _export()
    assert inline_set_grad_hops(exported.graph_module)
    with pytest.raises(Exception, match="User output|Input spec"):
        exported.validate()


def test_stale_source_slot_refused_before_signature_rebinding():
    exported, _ = _export()
    exported.graph_signature.output_specs[0].arg.name = "unrelated_same_shape"
    with pytest.raises(ValueError, match="exact graph value"):
        inline_set_grad_hops(exported.graph_module, graph_signature=exported.graph_signature)


def test_missing_tensor_metadata_is_not_a_type_proof():
    exported, _ = _export()
    node = next(node for node in exported.graph_module.graph.nodes if node.op == "placeholder")
    node.meta.pop("val")
    with pytest.raises(ValueError, match="tensor metadata"):
        inline_set_grad_hops(exported.graph_module, graph_signature=exported.graph_signature)


def test_output_slot_count_mismatch_refused():
    exported, _ = _export()
    exported.graph_signature.output_specs.pop()
    with pytest.raises(ValueError, match="slot count"):
        inline_set_grad_hops(exported.graph_module, graph_signature=exported.graph_signature)


def test_no_hops_does_not_modify_signature():
    class Plain(torch.nn.Module):
        def forward(self, x):
            return x + 1

    exported = torch.export.export(Plain(), (torch.ones(4),))
    signature = copy.deepcopy(exported.graph_signature)
    assert not inline_set_grad_hops(exported.graph_module, graph_signature=exported.graph_signature)
    assert exported.graph_signature == signature
    exported.validate()


def test_buffer_mutation_kind_and_target_are_preserved():
    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("total", torch.zeros(4))

        def forward(self, x):
            with torch.no_grad():
                self.total.add_(x)
                return self.total.sin()

    value = torch.arange(4, dtype=torch.float32)
    exported = torch.export.export(Buffered(), (value,))
    before_specs = [(spec.kind, spec.target) for spec in exported.graph_signature.output_specs]
    state = {key: tensor.clone() for key, tensor in exported.state_dict.items()}
    expected = exported.module()(value)
    with torch.no_grad():
        for key, tensor in state.items():
            exported.state_dict[key].copy_(tensor)
    assert inline_set_grad_hops(exported.graph_module, graph_signature=exported.graph_signature)
    exported.validate()
    actual = exported.module()(value)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert before_specs == [(spec.kind, spec.target) for spec in exported.graph_signature.output_specs]
    assert torch.equal(exported.state_dict["total"], value)


def test_input_slot_name_mismatch_refused():
    exported, _ = _export()
    exported.graph_signature.input_specs[0].arg.name = "unrelated_input"
    with pytest.raises(ValueError, match="exact graph value"):
        inline_set_grad_hops(exported.graph_module, graph_signature=exported.graph_signature)


def test_public_traced_conversion_accepts_verified_inlined_output():
    import m2m

    class TensorOutput(torch.nn.Module):
        @torch.no_grad()
        def forward(self, input):
            return input.sin()[:, 1:]

    result = m2m.convert(TensorOutput(), (torch.ones(3, 8),), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert result.module is not None
    assert not any("opaque" in str(op.name) for op in result.module.walk())
