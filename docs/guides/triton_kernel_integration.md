# Triton kernel integration (model2MLIR side)

A model can call a hand-written kernel, such as a `@torch.library.triton_op`, that model2MLIR should
not decompose. model2MLIR captures each call to a **registered** kernel as a typed `func.call` to a
body-less declaration tagged with the kernel's name. A downstream compiler later supplies the code.

This page covers the capture side. The other half, which links a body into the model, compiles it for
rv64gcv and runs it on spike, is merlin's
[Triton kernel integration guide](https://github.com/ucb-bar/merlin/blob/trition-op-integration/docs/guides/triton_kernel_integration.md)
(`docs/guides/triton_kernel_integration.md` on merlin's `trition-op-integration` branch).

- [Part 1: Summary](#part-1-summary): [what it does](#what-it-does), [how it is implemented](#how-it-is-implemented), [limitations](#limitations)
- [Part 2: How to use it](#part-2-how-to-use-it): [setup](#1-setup), [capture](#2-capture-the-example), [read the IR](#3-read-the-emitted-ir), [register a kernel](#4-register-a-new-kernel), [run downstream](#5-run-it-downstream-in-merlin), [tests](#6-tests)

---

# Part 1: Summary

## What it does

`KERNEL_REGISTRY` in `m2m/ir/kernel_registry.py` is a global table of custom ops that are kernels. It
sits alongside `DECOMPOSITION_TABLE`. When `torch.export` produces a node whose op is in the table, the
FX importer emits **one black-box call** for it instead of an opaque op:

- tensor inputs become SSA operands, in the op **schema's** argument order;
- every output becomes an SSA result, in return order;
- non-tensor arguments (scales, block sizes, omitted defaults) are recorded by name in an attribute;
- the call and its private declaration are tagged `prov.kernel = "<ns>::<name>"`.

model2MLIR never looks inside the kernel. Capture needs no GPU, and the kernel never runs during
export. Triton is only how the kernel is written; the consumer compiles it for its own target.

A registered kernel call is **not** opaque: `opaque_report` skips it, `kernel_report` counts it, and
`workloads/capture.py` prints `kernels=N`. Kernel-free models are unaffected. Their captures are
byte-identical to a capture without the registry.

### The IR contract

This is everything a consumer gets for a kernel:

| attribute | on | meaning |
|---|---|---|
| `prov.kernel` | call + declaration | identity, `"<ns>::<name>"`. The callee symbol is `<ns>__<name>`, with a `_<n>` suffix for each new shape, so match on this attribute, never on the symbol. |
| `prov.kernel_operands` | call | schema argument name of each SSA operand, in order |
| `prov.kernel_args` | call | non-tensor arguments by name (an absent optional is `unit`); **omitted** when there are none |

Kernel calls also get the provenance tags every op gets: `prov.op` and `prov.family = "kernel"`,
`prov.region_id`, `prov.aten` and `prov.orig_dtype`. Declarations are placed after `@forward`, so the
entry function is always the first `func.func`.

model2MLIR deliberately records nothing about where a kernel's code lives or what language it is in.
That belongs to the consumer's table. An early draft also emitted `prov.kernel_backend`,
`prov.triton_jit` and `prov.kernel_source`; merlin read none of them, so they were removed.

## How it is implemented

| piece | file | what it does |
|---|---|---|
| Registry | `m2m/ir/kernel_registry.py` | `KernelSpec(qualname, description)` entries in `_BUILTIN_KERNELS`. A qualname matches every overload of `torch.ops.<ns>.<name>`. `emit_kernel_call` builds the call from the op schema, and `kernel_decl` builds the private declaration. |
| Importer | `m2m/ir/import_fx.py` | `lookup_kernel` runs **before** the decomposition table and the result-type check, because a kernel may return a tuple. `_lower_kernel` takes result types from `meta["val"]`, gives one symbol to each (kernel, operand types, result types) combination, and resolves `getitem(node, i)` to result `i`. Declarations are collected separately and appended after `@forward`. A kernel inside a `while_loop` body keeps its `prov.kernel*` attributes. |
| Coverage | `m2m/coverage.py` | `opaque_report` excludes kernel calls. `kernel_report(text)` counts them by `prov.kernel`. `kernel` is in `CANONICAL_FAMILIES`. Both reports parse the module rather than matching text. |
| Backend choice | `m2m/api.py`, `m2m/capture/torch_export.py`, `m2m/capture/torch_mlir_bridge.py` | torch-mlir cannot legalize custom ops, so `backend="auto"` goes straight to the FX importer when the graph contains a registered kernel. The unsupported-op scan treats registered kernels as supported. |
| Transforms | `m2m/transforms/` | `split_by_section` now takes the first function **with a body** as the entry point and re-declares each section's callees. Before, it picked a declaration and returned no sections. `fuse_int2_unpack` no longer matches kernel calls by callee substring. |
| Workloads | `workloads/mlp_basic/`, `workloads/mlp_basic_triton/` | The same 3-layer MLP (16 → 32 → 32 → 8, same seed, weights and input), first in plain PyTorch, then with fc1+relu+fc2+relu as one Triton op, `torch.ops.triton_exp.fused_mlp2` (`kernels.py`). |

Errors are explicit. A registered kernel that cannot be represented faithfully, such as one that
mutates an argument or has an unresolved tensor operand, raises `KernelLoweringError`, and the importer
reports an error diagnostic instead of emitting wrong IR. An **unregistered** Triton op still lowers
through the generic opaque path, with a diagnostic that points at the registry.

## Limitations

| gap | today | what it needs |
|---|---|---|
| Kernels that write into their arguments (`mutates_args`, such as an `out=` buffer) | `torch.export` wraps them in `auto_functionalized_v2`. A registered one is an **import error**. An unregistered one takes the generic opaque path, which loses its operands without flagging it. | Rebuild the schema arguments, taking the mutated buffer from `_all_bases[...]`. Emit destination-passing style: the buffer's initial value becomes an operand, and the results are the op's returns plus one new value per mutated argument, tagged `prov.kernel_mutated` / `prov.kernel_tied`. Remap `getitem`. Refuse view arguments (`_<arg>_size` / `_stride`). Probe the `meta["val"]` slot layout for ops that both return and mutate first. |
| Unregistered custom ops | Generic opaque path: non-tensor arguments are dropped and only the first output is kept | Register them, or extend the same binding to the opaque path |
| One symbol per shape only | Two calls with the same shapes but different `prov.kernel_args` share a symbol | Key symbols by arguments too, once a consumer reads `kernel_args` |
| Registration | One global built-in table, with no per-workload registration | By design for now |
| Dynamic shapes | Dimensions become `-1` like any op's, but no consumer handles a dynamic kernel yet | A consumer that does |

The consumer has its own limits on what it can link: no `kernel_args`, a body per shape, and only
llvm-dialect bodies. They are listed in
[merlin's guide](https://github.com/ucb-bar/merlin/blob/trition-op-integration/docs/guides/triton_kernel_integration.md#limitations).

---

# Part 2: How to use it

## 1. Setup

```bash
uv venv && uv pip install -e .
uv pip install triton        # only to define and register the @triton_op; no GPU needed
```

Both MLP workloads use the repo's own venv (`capture.toml`: `venv = "../../.venv"`).

## 2. Capture the example

```bash
python workloads/capture.py mlp_basic_triton --formats fp32
python workloads/capture.py mlp_basic        --formats fp32
```

```
OK mlp_basic_triton fp32  opaque=0    kernels=1   linalg=5     families=4   -> workloads/mlp_basic_triton/mlp_basic_triton.mlir
OK mlp_basic      fp32  opaque=0    kernels=0   linalg=19    families=4   -> workloads/mlp_basic/mlp_basic.mlir
```

Each run writes `workloads/<name>/<name>.mlir`, plus `<name>.safetensors` and its `.manifest.json`,
which hold the weights keyed by function-argument index. These are gitignored build outputs.

The same capture from Python:

```python
import sys; sys.path.insert(0, "workloads/mlp_basic_triton")
from loader import get_model_and_inputs
import m2m
from m2m.coverage import kernel_report, opaque_report

model, inputs = get_model_and_inputs()
r = m2m.convert(model, inputs)        # backend="auto"
r.path_taken                          # "fx_importer": torch-mlir is skipped when a kernel is present
opaque_report(r.mlir_text)            # {}
kernel_report(r.mlir_text)            # {"triton_exp::fused_mlp2": 1}
```

## 3. Read the emitted IR

`workloads/mlp_basic_triton/mlp_basic_triton.mlir`:

```mlir
builtin.module attributes {prov.weights_file = ".../mlp_basic_triton.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x16xf32>, %1: tensor<32xf32>, %2: tensor<32x32xf32>, %3: tensor<32xf32>,
                     %4: tensor<8x32xf32>, %5: tensor<8xf32>, %6: tensor<4x16xf32>) -> tensor<4x8xf32> {
    %7 = func.call @triton_exp__fused_mlp2(%6, %0, %1, %2, %3) {prov.kernel = "triton_exp::fused_mlp2",
           prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"], prov.region_id = "triton_exp__fused_mlp2_0",
           prov.op = "kernel", prov.family = "kernel", prov.aten = "triton_exp.fused_mlp2.default", ...}
         : (tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
    ... fc3: linalg.transpose / linalg.fill / linalg.matmul / bias linalg.generic, identical to mlp_basic ...
    func.return %15 : tensor<4x8xf32>
  }
  func.func private @triton_exp__fused_mlp2(tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>,
                                            tensor<32xf32>) -> tensor<4x32xf32> attributes {prov.kernel = "triton_exp::fused_mlp2"}
}
```

Compared with `mlp_basic.mlir`, the `@forward` signature is the same (7 arguments in the same order),
the weights manifest is the same, and fc3 is identical op for op. Only fc1..relu2 is replaced, by the
one call.

## 4. Register a new kernel

1. Define the op as usual, for example `@triton_op("my_ns::my_kernel", mutates_args={})`. It must
   return its outputs rather than write into an argument (see [Limitations](#limitations)).
2. Append a `KernelSpec` to `_BUILTIN_KERNELS` in `m2m/ir/kernel_registry.py`:

   ```python
   KernelSpec("my_ns::my_kernel", description="what it computes, which workload uses it"),
   ```

3. Capture a model that calls it and check `kernel_report(r.mlir_text)` and `opaque_report`.
4. Add the matching entry and body on the consumer side. In merlin, that is
   `src/merlin/llvmlower/kernel_table.py`, plus a body under `src/merlin/kernels/triton_llvmir/`.
   merlin's `test_table_keys_match_m2m_registry` fails until the two tables agree.

## 5. Run it downstream in merlin

merlin consumes this IR as is. Its kernel table maps `triton_exp::fused_mlp2` to the Triton kernel's
llvm-dialect MLIR, links that into the model **before** MLIR lowering, and runs the result on spike
rv64gcv. Measured against PyTorch:

| capture | spike retired instructions | max abs error |
|---|---|---|
| `mlp_basic_triton` (kernel linked) | 15,121 | 3.0e-8 |
| `mlp_basic` (no kernel) | 16,156 | 4.5e-8 |

merlin tracks copies of both captures under `merlin/tests/data/mlp_bundles/`, and
`merlin/tests/infra/test_mlp_bundles_fresh.py` re-captures them from this checkout. To regenerate
merlin's copies and run them there, follow
[merlin's guide](https://github.com/ucb-bar/merlin/blob/trition-op-integration/docs/guides/triton_kernel_integration.md#part-2-how-to-use-it).

## 6. Tests

```bash
python -m pytest tests/test_kernels.py
```

There are 13 tests. Most register plain CPU `torch.library.custom_op`s with fake implementations as
kernels, so they need neither Triton nor a GPU. `test_mlp_basic_triton_mvp` captures the real
workload and needs the `triton` package.
