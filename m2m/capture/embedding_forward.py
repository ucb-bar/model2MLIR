"""Closed unary embedding wrapper admission, not arbitrary Python verification."""

from __future__ import annotations
import __future__

import ast
import builtins
import copy
import hashlib
import inspect
import json
import math
import struct
import textwrap
import tokenize
from collections import OrderedDict
from dataclasses import dataclass
from types import CodeType, FunctionType

import torch

_LOOKUP = tuple(
    (n, inspect.getattr_static(torch.nn.Module, n))
    for n in (
        "__getattribute__",
        "__getattr__",
        "__dict__",
        "__call__",
        "_call_impl",
        "_wrapped_call_impl",
    )
)
_PUBLIC_FORWARD = torch.nn.Embedding.forward


def _error(message):
    from m2m.capture.dynamic_linear_trace import DynamicLinearLineageError

    raise DynamicLinearLineageError("unproved pure embedding wrapper: " + message)


def _dictionary(module):
    if type(type(module)) is not type:
        _error("custom metaclass")
    if any(inspect.getattr_static(type(module), n) is not fn for n, fn in _LOOKUP):
        _error("custom module lookup or descriptor")
    state = object.__getattribute__(module, "__dict__")
    if type(state) is not dict or any(type(k) is not str for k in state):
        _error("module dictionary")
    if (
        state.get("_compiled_call_impl") is not None
        or inspect.getattr_static(type(module), "_compiled_call_impl") is not None
    ):
        _error("compiled call override")
    for name in ("_parameters", "_buffers", "_modules"):
        if type(state.get(name)) is not dict:
            _error("module state container")
    for name in (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
    ):
        value = state.get(name)
        if type(value) not in (dict, OrderedDict) or value:
            _error("module hooks")
    from torch.nn.modules import module as public_module

    for name in (
        "_global_forward_hooks",
        "_global_forward_pre_hooks",
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
    ):
        value = getattr(public_module, name)
        if type(value) not in (dict, OrderedDict) or value:
            _error("global module hooks")
    return state


def _member(module, state, name):
    # Reject descriptors before any getattr can execute them. Only ordinary
    # instance scalars and actual registered parameters/buffers are read.
    if any(name in cls.__dict__ for cls in type(module).__mro__):
        _error("class attribute or descriptor: " + name)
    if name in state:
        return "instance", state[name]
    for container in ("_parameters", "_buffers"):
        if name in state[container]:
            return container, state[container][name]
    _error("unowned attribute: " + name)


def _attribute(node, root, name=None):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == root
        and (name is None or node.attr == name)
    )


def _code_parts(code):
    from m2m.capture.embedding_trace import frozen_metadata

    return (
        code.co_code,
        frozen_metadata(code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_freevars,
        code.co_cellvars,
        code.co_flags,
        code.co_stacksize,
        code.co_nlocals,
        getattr(code, "co_exceptiontable", b"").hex(),
    )


def _source_context(fn):
    """Recover compiler context from physical source, never candidate code flags."""
    path = inspect.getsourcefile(fn)
    if path is None:
        _error("physical source context unavailable")
    try:
        with tokenize.open(path) as stream:
            source = stream.read(1_048_577)
        if len(source) > 1_048_576:
            _error("physical source context exceeds proof bound")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeError) as exc:
        _error("physical source context: " + str(exc))
    flags = 0
    for statement in tree.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) is str
        ):
            continue
        if (
            not isinstance(statement, ast.ImportFrom)
            or statement.module != "__future__"
        ):
            break
        for alias in statement.names:
            if alias.name not in __future__.all_feature_names:
                _error("unproved future context")
            flags |= getattr(__future__, alias.name).compiler_flag
    found = []

    def visit(node, depth):
        function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if (
            function
            and node.lineno == fn.__code__.co_firstlineno
            and node.name == fn.__code__.co_name
        ):
            found.append(depth > 0)
        for child in ast.iter_child_nodes(node):
            visit(child, depth + int(function))

    visit(tree, 0)
    if len(found) != 1:
        _error("unproved source definition/qualification context")
    return flags, found[0]


def _grammar(fn, module):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError) as exc:
        _error("source unavailable: " + str(exc))
    if (
        len(list(ast.walk(tree))) > 128
        or len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
    ):
        _error("forward definition")
    definition = tree.body[0]
    args = definition.args
    positional = args.posonlyargs + args.args
    if (
        len(positional) != 2
        or args.vararg
        or args.kwarg
        or args.kwonlyargs
        or args.defaults
        or fn.__defaults__ is not None
        or fn.__kwdefaults__ is not None
        or definition.decorator_list
    ):
        _error("unary signature/defaults")
    body = definition.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and type(body[0].value.value) is str
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        _error("only a pure return expression is supported")
    self_name, input_name = (x.arg for x in positional)
    value = body[0].value
    scale = None
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
        value, scale = value.left, value.right
    if not (
        isinstance(value, ast.Call)
        and not value.keywords
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == input_name
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "forward"
        and isinstance(value.func.value, ast.Call)
        and not value.func.value.args
        and not value.func.value.keywords
        and isinstance(value.func.value.func, ast.Name)
        and value.func.value.func.id == "super"
    ):
        _error("lookup is not the original public super().forward(input)")
    if (
        type(fn.__globals__) is not dict
        or type(fn.__builtins__) is not dict
        or fn.__globals__.get("super", builtins.super) is not builtins.super
        or fn.__builtins__.get("super") is not builtins.super
    ):
        _error("super binding")
    if (
        fn.__code__.co_freevars != ("__class__",)
        or not fn.__closure__
        or len(fn.__closure__) != 1
    ):
        _error("closure")
    owner = fn.__closure__[0].cell_contents
    mro = type(module).__mro__
    if not isinstance(owner, type) or owner not in mro:
        _error("super class owner")
    target = next(
        (
            cls.__dict__["forward"]
            for cls in mro[mro.index(owner) + 1 :]
            if "forward" in cls.__dict__
        ),
        None,
    )
    if target is not _PUBLIC_FORWARD:
        _error("super target is not public embedding")
    to_dtype = False
    if isinstance(scale, ast.Call):
        if (
            scale.keywords
            or len(scale.args) != 1
            or not isinstance(scale.func, ast.Attribute)
            or scale.func.attr != "to"
            or not isinstance(scale.args[0], ast.Attribute)
            or scale.args[0].attr != "dtype"
            or not _attribute(scale.args[0].value, self_name, "weight")
        ):
            _error("only buffer.to(self.weight.dtype) is supported")
        scale, to_dtype = scale.func.value, True
    if scale is not None and not _attribute(scale, self_name):
        _error("scale is not an ordinary owned attribute")
    # Compile but NEVER execute the source. This proves source text describes
    # the actual code object, including names/constants/closure and operands.
    source_flags, nested = _source_context(fn)
    definition_class = ast.ClassDef(
        name="_Proof",
        bases=[],
        keywords=[],
        body=[copy.deepcopy(definition)],
        decorator_list=[],
    )
    body = [definition_class]
    if nested:
        body = [
            ast.FunctionDef(
                name="_LexicalContext",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=body,
                decorator_list=[],
            )
        ]
    template = ast.Module(body=body, type_ignores=[])
    module_code = compile(
        ast.fix_missing_locations(template),
        "<embedding-forward-proof>",
        "exec",
        flags=source_flags,
        dont_inherit=True,
    )
    context_code = next(c for c in module_code.co_consts if type(c) is CodeType)
    class_code = (
        next(c for c in context_code.co_consts if type(c) is CodeType)
        if nested
        else context_code
    )
    code = next(c for c in class_code.co_consts if type(c) is CodeType)
    if _code_parts(fn.__code__) != _code_parts(code):
        _error("source/code mismatch")
    return owner, None if scale is None else scale.attr, to_dtype


@dataclass(frozen=True, slots=True)
class ForwardContract:
    module_type: type
    function: object
    code: object
    closure_owner: object
    scale_name: str | None
    scale_kind: str | None
    scalar_bits: tuple | None
    scale: object
    scale_seal: object


def observe(module, *, wrapper=None):
    state = _dictionary(module)
    if "forward" in state and state["forward"] is not wrapper:
        _error("instance forward override")
    fn = inspect.getattr_static(type(module), "forward")
    if type(fn) is not FunctionType or fn.__dict__:
        _error("forward callable")
    for name in (
        "weight",
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
        "training",
    ):
        _member(module, state, name)
    owner, name, to_dtype = (
        (None, None, False) if fn is _PUBLIC_FORWARD else _grammar(fn, module)
    )
    kind, value, bits, seal = None, None, None, None
    if name is not None:
        kind, value = _member(module, state, name)
        if (
            kind == "instance"
            and type(value) in (int, float)
            and math.isfinite(value)
            and not to_dtype
        ):
            bits = (
                type(value).__name__,
                value if type(value) is int else struct.pack(">d", value).hex(),
            )
        elif (
            kind == "_buffers"
            and type(value) is torch.Tensor
            and value.dtype is torch.float32
            and value.shape == ()
            and not value.requires_grad
            and not any(
                n in value.__dict__
                for n in ("to", "__torch_function__", "__torch_dispatch__")
            )
        ):
            from m2m.capture.dynamic_linear_trace import _seal

            seal = _seal(name, value)
        else:
            _error("scale is not a finite builtin scalar or ordinary F32 scalar buffer")
    return ForwardContract(
        type(module), fn, fn.__code__, owner, name, kind, bits, value, seal
    )


def validate(module, contract, *, wrapper):
    if type(contract) is not ForwardContract:
        _error("forward contract owner")
    current = observe(module, wrapper=wrapper)
    if (
        current.module_type is not contract.module_type
        or current.function is not contract.function
        or current.code is not contract.code
        or current.closure_owner is not contract.closure_owner
        or current.scale_name != contract.scale_name
        or current.scale_kind != contract.scale_kind
        or current.scalar_bits != contract.scalar_bits
    ):
        _error("forward/code/closure/scalar changed")
    if contract.scale_seal is not None:
        from m2m.capture.dynamic_linear_trace import _check_seal

        _check_seal(current.scale, contract.scale_seal)


def summary(contract):
    parts = _code_parts(contract.code)
    return {
        "kind": "public_lookup_or_closed_unary_postscale_v1",
        "source_function_module": contract.function.__module__,
        "source_function_qualname": contract.function.__qualname__,
        "executed_code_sha256": hashlib.sha256(
            json.dumps([parts[0].hex(), *parts[1:]], sort_keys=True).encode()
        ).hexdigest(),
        "actual_source_code_compared": contract.function is not _PUBLIC_FORWARD,
        "scale_attribute": contract.scale_name,
        "scale_owner_kind": contract.scale_kind,
        "scalar_bits": contract.scalar_bits,
        "buffer_sha256": None
        if contract.scale_seal is None
        else contract.scale_seal.sha256,
        "arbitrary_python_effect_proof": False,
    }
