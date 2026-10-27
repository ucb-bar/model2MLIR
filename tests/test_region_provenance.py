"""Call-bound normalization receipts must survive export and PT2E."""
import copy

import pytest
import torch

from m2m.capture.trace import capture_frontend_snapshot, graph_relation, materialize_frontend_precision


class Nested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        with torch.no_grad():
            a = self.region(x)
        return self.fc(a)

    def region(self, x):
        a = x.unsqueeze(0).float()
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
            a = a @ a.transpose(1, 2)
            a = a.cos() * 1.
        return a


def staged_graphs():
    inputs = (torch.randn(4, 4),)
    model = Nested().eval()
    gm, args, original, receipt = materialize_frontend_precision(
        model, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    staged = capture_frontend_snapshot(gm, args, stage='staged')
    assert torch.equal(model(*inputs), gm(*args))
    return gm, args, original, staged, receipt


def assert_complete(source, dest):
    relation = graph_relation(source, dest)
    assert relation['unresolved_source_ids'] == [], relation
    assert relation['unresolved_destination_ids'] == [], relation


def test_admitted_nested_region_survives_repeated_export():
    gm, args, original, staged, receipt = staged_graphs()
    assert_complete(original, staged)
    for stage in ('quantization_input', 'quantized'):
        program = torch.export.export(gm, args)
        observed = capture_frontend_snapshot(program, stage=stage)
        assert_complete(original, observed)
        assert_complete(staged, observed)
        gm = program.module()
    row, = receipt['inlined_autocast_regions']
    assert row['operand_bindings'] and row['output_bindings'] and row['ownership_moves']
    original_nodes = {node['id']: node for node in original['nodes']}
    removed = {identity for item in row['eliminations'] for identity in item['source_ids']}
    assert {'wrap_with_autocast', '<built-in function getitem>'} <= {
        original_nodes[identity]['target'] for identity in removed}


def test_plain_grad_disabled_single_tuple_selection():
    class Plain(torch.nn.Module):
        @torch.no_grad()
        def forward(self, x):
            return x.unsqueeze(0).cos()
    gm, args, original, _ = materialize_frontend_precision(
        Plain(), (torch.randn(4, 4),), dtype=torch.float32, retarget_float_dtype_arguments=True)
    assert_complete(original, capture_frontend_snapshot(gm, args, stage='staged'))


def test_nested_region_survives_actual_pt2e():
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    gm, args, original, staged, _ = staged_graphs()
    quantized = apply_quantization(
        gm, QuantizationConfig(scheme='int8_static_act_int8_weight'),
        example_inputs=args, original_frontend_snapshot=original)
    observed = capture_frontend_snapshot(quantized, args, stage='quantized')
    assert_complete(original, observed)
    assert_complete(staged, observed)
    inner = [node for node in observed['nodes'] if node['graph_id'] != 'g:quantized:root'
             and node['op'] == 'call_function']
    wrapper_ids = {node['id'] for node in original['nodes']
                   if node['target'] == 'wrap_with_set_grad_enabled'}
    assert inner
    assert all(not wrapper_ids.intersection(node['origin_node_ids']) for node in inner)


@pytest.mark.parametrize('change', [
    'caller_operand', 'caller_result', 'caller_ambiguous', 'reused_body',
    'enabled_grad', 'unknown_wrapper', 'source_enabled_autocast',
    'source_output_order', 'destination_output_order', 'nonidentity_cast',
])
def test_receipt_refuses_changed_region(change):
    _, _, source, dest, _ = staged_graphs()
    source, dest = copy.deepcopy(source), copy.deepcopy(dest)
    sw = next(node for node in source['nodes'] if node['target'] == 'wrap_with_set_grad_enabled')
    dw = next(node for node in dest['nodes'] if node['target'] == 'wrap_with_set_grad_enabled')
    nested = next(node for node in source['nodes'] if node['target'] == 'aten.cos.default')
    if change == 'caller_operand':
        dw['args'][2] = dw['args'][1]
    elif change == 'caller_result':
        dw['args'][2]['value_id'] = 'unbound-result'
    elif change == 'caller_ambiguous':
        # A second destination with the same caller origin is not a unique binding.
        operand = next(node for node in dest['nodes'] if node['id'] == dw['args'][2]['node_id'])
        other = copy.deepcopy(operand)
        other['id'] += ':duplicate'
        dest['nodes'].append(other)
    elif change == 'reused_body':
        other = copy.deepcopy(dw)
        other['id'] += ':duplicate'
        dest['nodes'].append(other)
    elif change == 'enabled_grad':
        dw['args'][0] = True
    elif change == 'unknown_wrapper':
        dw['target'] = 'unrecognized_region'
    elif change == 'source_enabled_autocast':
        next(node for node in source['nodes'] if node['target'] == 'wrap_with_autocast')['args'][2] = True
    elif change == 'source_output_order':
        output = next(node for node in source['nodes'] if node['op'] == 'output'
                      and node['graph_id'] == nested['graph_id'])
        output['args'][0] = [output['args'][0][0], output['args'][0][0]]
    elif change == 'destination_output_order':
        output = next(node for node in dest['nodes'] if node['op'] == 'output'
                      and node['graph_id'] != 'g:staged:root')
        output['args'][0] = [output['args'][0][0], output['args'][0][0]]
    elif change == 'nonidentity_cast':
        cast = next(node for node in source['nodes'] if node['target'] == 'aten.to.dtype')
        cast['kwargs']['copy'] = True
    relation = graph_relation(source, dest)
    assert relation['status'] == 'diagnostic'
    assert nested['id'] in relation['unresolved_source_ids'], relation


def test_committed_repro_resolves_through_public_admission():
    """The saved TinyLlama-shaped repro: a no-grad region holding a disabled-autocast body with a metadata
    assertion is admitted by the public materializer and keeps complete lineage through every re-export."""
    class Committed(torch.nn.Module):
        @torch.no_grad()
        def forward(self, x):
            a = x.unsqueeze(0).float()
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                a = a.float() @ a.float().transpose(1, 2)
                a = a.cos() * 1.
            return a.to(x.dtype)

    inputs = (torch.randn(4, 4),)
    model = Committed()
    gm, args, original, receipt = materialize_frontend_precision(
        model, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    assert torch.equal(model(*inputs), gm(*args))
    snapshots = [original]
    for stage in ('staged', 'quantization_input', 'quantized'):
        exported = torch.export.export(gm, args)
        snapshots.append(capture_frontend_snapshot(exported, stage=stage))
        gm = exported.module()
    for source, dest in zip(snapshots, snapshots[1:]):
        assert_complete(source, dest)
    assert_complete(original, snapshots[-1])
    assert receipt['staged_precision_audit']['non_target_floating_values'] == 0


@pytest.mark.parametrize('side', ['source', 'destination'])
def test_equal_typed_output_permutation_is_refused(side):
    class Pair(torch.nn.Module):
        @torch.no_grad()
        def forward(self, x):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                return x.cos(), x.sin()

    gm, args, source, _ = materialize_frontend_precision(
        Pair(), (torch.randn(4, 4),), dtype=torch.float32, retarget_float_dtype_arguments=True)
    dest = capture_frontend_snapshot(gm, args, stage='staged')
    assert_complete(source, dest)
    altered = copy.deepcopy(source if side == 'source' else dest)
    output = next(node for node in altered['nodes'] if node['op'] == 'output'
                  and len(node['args'][0]) == 2 and node['graph_id'] != altered['graph_id'])
    output['args'][0].reverse()
    relation = graph_relation(altered, dest) if side == 'source' else graph_relation(source, altered)
    assert relation['status'] == 'diagnostic'


@pytest.mark.parametrize('change', ['source_reuse', 'unknown_autocast', 'enabled_autocast', 'guard_dtype'])
def test_original_normalization_contract_cannot_be_reinterpreted(change):
    _, _, source, dest, _ = staged_graphs()
    source = copy.deepcopy(source)
    nested = next(node for node in source['nodes'] if node['target'] == 'aten.cos.default')
    wrapper = next(node for node in source['nodes'] if node['target'] == 'wrap_with_autocast')
    if change == 'source_reuse':
        duplicate = copy.deepcopy(wrapper)
        duplicate['id'] += ':second_call'
        source['nodes'].append(duplicate)
    elif change == 'unknown_autocast':
        wrapper['args'][3] = 'unknown'
    elif change == 'enabled_autocast':
        wrapper['args'][2] = True
    elif change == 'guard_dtype':
        guard = next(node for node in source['nodes']
                     if node['target'] == 'aten._assert_tensor_metadata.default')
        guard['kwargs']['dtype'] = {'kind': 'dtype', 'value': 'torch.float16'}
    relation = graph_relation(source, dest)
    assert relation['status'] == 'diagnostic'
    assert nested['id'] in relation['unresolved_source_ids']


def test_admitted_region_has_complete_conversion_trace():
    import m2m
    from m2m.capture.torchao_pipeline import QuantizationConfig

    gm, args, original, _, _ = staged_graphs()
    result = m2m.convert(gm, args, backend='fx_importer', capture_trace=True,
        original_frontend_snapshot=original,
        quantization=QuantizationConfig(scheme='int8_static_act_int8_weight'))
    assert result.ok, result.diagnostics
    assert result.capture_trace['status'] == 'complete', result.capture_trace['blockers']


def test_single_result_selection_requires_observed_tuple_abi():
    _, _, source, dest, _ = staged_graphs()
    for snapshot in (source, dest):
        wrapper = next(node for node in snapshot['nodes']
                       if node['target'] == 'wrap_with_set_grad_enabled')
        wrapper['result_metadata']['container'] = 'single'
    relation = graph_relation(source, dest)
    selector = next(node for node in source['nodes']
                    if node['target'] == '<built-in function getitem>'
                    and node['graph_id'] == source['graph_id'])
    assert selector['id'] in relation['unresolved_source_ids']


@pytest.mark.parametrize('change', ['caller', 'enabled', 'unknown', 'output_order', 'reuse'])
def test_intermediate_stage_tags_do_not_bypass_region_proof(change):
    gm, args, _, source, _ = staged_graphs()
    dest = capture_frontend_snapshot(gm, args, stage='quantization_input')
    wrapper = next(node for node in dest['nodes'] if node['target'] == 'wrap_with_set_grad_enabled')
    if change == 'caller':
        wrapper['args'][2] = wrapper['args'][1]
    elif change == 'enabled':
        wrapper['args'][0] = True
    elif change == 'unknown':
        wrapper['target'] = 'unknown_region'
    elif change == 'output_order':
        output = next(node for node in dest['nodes'] if node['op'] == 'output'
                      and node['graph_id'] != dest['graph_id'])
        output['args'][0] = [next(node for node in dest['nodes']
                                 if node['target'] == 'aten.cos.default')['args'][0]]
    elif change == 'reuse':
        duplicate = copy.deepcopy(wrapper)
        duplicate['id'] += ':reuse'
        dest['nodes'].append(duplicate)
    assert graph_relation(source, dest)['status'] == 'diagnostic'


@pytest.mark.parametrize('change', [
    'removed', 'missing_field', 'wrong_version', 'wrong_types', 'conflicting_rows',
    'operator_after_removal', 'removed_both', 'legacy_removed', 'missing_regions', 'bad_node',
    'bad_result', 'bad_region', 'bad_entries', 'bad_entry', 'changed_certificate',
])
@pytest.mark.parametrize('source_stage', ['original', 'staged'])
def test_certificate_is_required_and_schema_checked(change, source_stage):
    gm, args, original, staged, _ = staged_graphs()
    source = original if source_stage == 'original' else staged
    dest = capture_frontend_snapshot(gm, args, stage='quantization_input')
    if change in {'removed', 'operator_after_removal', 'removed_both', 'legacy_removed'}:
        dest.pop('nested_lineage')
        if change in {'removed_both', 'legacy_removed'}:
            source.pop('nested_lineage', None)
        if change == 'legacy_removed':
            # Legacy snapshots retain moved ancestry but lack the new marker.
            for snapshot in (source, dest):
                for node in snapshot['nodes']:
                    node.pop('nested_lineage_required', None)
        if change == 'operator_after_removal':
            next(node for node in dest['nodes'] if node['target'] == 'aten.cos.default')['target'] = 'aten.sin.default'
    else:
        row = dest['nested_lineage'][0]['receipt']
        if change == 'missing_field':
            row.pop('normalized_body')
        elif change == 'wrong_version':
            row['schema'] = 'm2m.nested_lineage.v0'
        elif change == 'wrong_types':
            row['bindings'] = ['invalid']
        elif change == 'missing_regions':
            row.pop('regions')
        elif change == 'bad_node':
            row['normalized_body'][0]['args'] = {}
        elif change == 'bad_result':
            row['normalized_body'][0]['results'][0]['shape'] = 7
        elif change == 'bad_region':
            row['regions'][0]['output_bindings'] = None
        elif change == 'bad_entries':
            dest['nested_lineage'] = {}
        elif change == 'bad_entry':
            dest['nested_lineage'][0] = None
        elif change == 'changed_certificate':
            row['torch_version'] = 'different-runtime'
        elif change == 'conflicting_rows':
            duplicate = copy.deepcopy(dest['nested_lineage'][0])
            duplicate['receipt']['lineages'][0] = ['conflicting-origin']
            dest['nested_lineage'].append(duplicate)
    relation = graph_relation(source, dest)
    assert relation['status'] == 'diagnostic', relation
    assert next(node['id'] for node in source['nodes']
                if node['target'] == 'aten.cos.default') in relation['unresolved_source_ids']


def test_wrong_tuple_selector_index_is_refused():
    class Pair(torch.nn.Module):
        @torch.no_grad()
        def forward(self, x):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                return x.cos(), x.sin()
    gm, args, _, _ = materialize_frontend_precision(
        Pair(), (torch.randn(4, 4),), dtype=torch.float32, retarget_float_dtype_arguments=True)
    source = capture_frontend_snapshot(gm, args, stage='staged')
    dest = capture_frontend_snapshot(gm, args, stage='quantization_input')
    selector = next(node for node in dest['nodes']
                    if node['target'] == '<built-in function getitem>' and node['args'][1] == 0)
    selector['args'][1] = 1
    parent = next(node for node in dest['nodes'] if node['id'] == selector['args'][0]['node_id'])
    selector['args'][0]['value_id'] = parent['results'][1]['id']
    relation = graph_relation(source, dest)
    assert relation['status'] == 'diagnostic', relation
    assert selector['id'] in relation['unresolved_destination_ids']


def test_unrelated_cond_branch_keeps_direct_ancestry():
    class Alongside(torch.nn.Module):
        def forward(self, x, pred):
            with torch.no_grad():
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                    a = x.cos()
            b = torch.cond(pred, lambda t: t.sin(), lambda t: t.exp(), (x,))
            return a + b
    gm, args, source, _ = materialize_frontend_precision(
        Alongside(), (torch.randn(4, 4), torch.tensor(True)),
        dtype=torch.float32, retarget_float_dtype_arguments=True)
    dest = capture_frontend_snapshot(gm, args, stage='staged')
    unrelated = {node['id'] for node in source['nodes']
                 if node['target'] in {'aten.sin.default', 'aten.exp.default'}}
    assert len(unrelated) == 2
    relation = graph_relation(source, dest)
    assert unrelated.isdisjoint(relation['unresolved_source_ids']), relation
    assert_complete(source, dest)


def test_plain_grad_call_has_no_certificate():
    class Plain(torch.nn.Module):
        @torch.no_grad()
        def forward(self, x):
            return x.cos()
    gm, args, source, _ = materialize_frontend_precision(
        Plain(), (torch.randn(4, 4),), dtype=torch.float32, retarget_float_dtype_arguments=True)
    dest = capture_frontend_snapshot(gm, args, stage='staged')
    assert not dest.get('nested_lineage')
    assert_complete(source, dest)


@pytest.mark.parametrize('change', ['operand_swap', 'roundtrip_cast', 'shared_body'])
def test_actual_fx_region_changes_are_refused(change):
    class Binary(torch.nn.Module):
        @torch.no_grad()
        def forward(self, x, y):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                return x - y
    gm, args, _, _ = materialize_frontend_precision(
        Binary(), (torch.randn(4, 4), torch.randn(4, 4)),
        dtype=torch.float32, retarget_float_dtype_arguments=True)
    source = capture_frontend_snapshot(gm, args, stage='staged')
    wrapper = next(node for node in gm.graph.nodes
                   if node.target is torch.ops.higher_order.wrap_with_set_grad_enabled)
    body = gm.get_submodule(str(wrapper.args[1].target))
    binary = next(node for node in body.graph.nodes if node.target is torch.ops.aten.sub.Tensor)
    if change == 'operand_swap':
        binary.args = (binary.args[1], binary.args[0])
    elif change == 'roundtrip_cast':
        operand = binary.args[0]
        with body.graph.inserting_before(binary):
            low = body.graph.call_function(torch.ops.aten.to.dtype, (operand, torch.bfloat16))
            high = body.graph.call_function(torch.ops.aten.to.dtype, (low, torch.float32))
        low.meta['val'] = operand.meta['val'].to(torch.bfloat16)
        high.meta['val'] = operand.meta['val'].to(torch.float32)
        binary.args = (high, binary.args[1])
    else:
        # Two real call sites refer to the same registered FX body.
        with gm.graph.inserting_after(wrapper):
            second = gm.graph.call_function(wrapper.target, wrapper.args, wrapper.kwargs)
        second.meta = dict(wrapper.meta)
        second.meta['custom'] = copy.deepcopy(wrapper.meta['custom'])
    body.graph.lint()
    body.recompile()
    gm.graph.lint()
    gm.recompile()
    # Observe this exact FX graph so export cannot remove the unused shared call.
    from types import SimpleNamespace
    from m2m.capture.trace import snapshot_exported_program
    dest = snapshot_exported_program(SimpleNamespace(graph_module=gm), stage='quantization_input')
    relation = graph_relation(source, dest)
    assert relation['status'] == 'diagnostic', relation
    assert next(node['id'] for node in source['nodes']
                if node['target'] == 'aten.sub.Tensor') in relation['unresolved_source_ids']


@pytest.mark.parametrize('nested', [False, True])
def test_multiple_certified_regions_fail_closed(nested):
    class Multiple(torch.nn.Module):
        def forward(self, x):
            with torch.no_grad():
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                    a = x.cos()
                if nested:
                    with torch.no_grad():
                        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                            a = a.sin()
            if not nested:
                with torch.no_grad():
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                        a = a.sin()
            return a
    gm, args, source, receipt = materialize_frontend_precision(
        Multiple(), (torch.randn(4, 4),), dtype=torch.float32, retarget_float_dtype_arguments=True)
    assert len(receipt['inlined_autocast_regions']) == 2
    dest = capture_frontend_snapshot(gm, args, stage='staged')
    relation = graph_relation(source, dest)
    # Known limitation: correspondence across multiple/nested normalized regions
    # is incomplete; never turn ambiguous inherited identities into a proof.
    assert relation['status'] == 'diagnostic', relation
    assert relation['unresolved_source_ids'] or relation['unresolved_destination_ids']


@pytest.mark.parametrize('change', ['missing_field', 'wrong_version', 'wrong_types', 'conflicting_copy'])
def test_fx_certificate_reader_refuses_malformed_metadata(change):
    from types import SimpleNamespace
    from m2m.capture.region_lineage import KEY
    from m2m.capture.trace import snapshot_exported_program

    gm, args, _, source, _ = staged_graphs()
    wrapper = next(node for node in gm.graph.nodes
                   if node.target is torch.ops.higher_order.wrap_with_set_grad_enabled)
    wrapper.meta['custom'] = copy.deepcopy(wrapper.meta['custom'])
    row = wrapper.meta['custom'][KEY]
    if change == 'missing_field':
        row.pop('regions')
    elif change == 'wrong_version':
        row['schema'] = 'm2m.nested_lineage.v0'
    elif change == 'wrong_types':
        row['lineages'] = 1
    else:
        selector = next(iter(wrapper.users))
        selector.meta['custom'] = dict(selector.meta.get('custom') or {})
        selector.meta['custom'][KEY] = None
    # The public snapshot implementation's restore step must also avoid raising.
    dest = snapshot_exported_program(SimpleNamespace(graph_module=gm), stage='quantization_input')
    relation = graph_relation(source, dest)
    assert relation['status'] == 'diagnostic'
    assert relation['unresolved_source_ids']


@pytest.mark.parametrize('change', ['removed', 'operator', 'wrong_version', 'wrong_types', 'source_operand'])
def test_flattened_transition_requires_its_exact_certificate(change):
    import m2m

    gm, args, original, _, _ = staged_graphs()
    result = m2m.convert(gm, args, backend='fx_importer', capture_trace=True,
                         original_frontend_snapshot=original)
    assert result.ok, result.diagnostics
    assert result.capture_trace['status'] == 'complete', result.capture_trace['blockers']
    source = result.capture_trace['graphs']['quantized']
    dest = copy.deepcopy(result.capture_trace['graphs']['prepared'])
    if change == 'removed':
        dest.pop('flattened_nested_lineage', None)
    elif change == 'operator':
        next(node for node in dest['nodes'] if node['target'] == 'aten.cos.default')['target'] = 'aten.sin.default'
    elif change == 'source_operand':
        wrapper = next(node for node in source['nodes'] if node['target'] == 'wrap_with_set_grad_enabled')
        wrapper['args'][2] = wrapper['args'][1]
    elif change == 'wrong_version':
        dest.setdefault('flattened_nested_lineage', {})['schema'] = 'm2m.flattened_nested_lineage.v0'
    else:
        dest.setdefault('flattened_nested_lineage', {})['destination_nodes'] = None
    relation = graph_relation(source, dest)
    assert relation['status'] == 'diagnostic', relation
    assert relation['unresolved_source_ids']
