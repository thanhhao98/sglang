import ast
import pathlib
import unittest


MODEL_PATH = pathlib.Path(
    "/Users/zhaol/projects/htphan-sglang/python/sglang/srt/models/qwen2.py"
)


def _load_module_ast():
    return ast.parse(MODEL_PATH.read_text())


def _find_class(module_ast, class_name: str) -> ast.ClassDef:
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found")


def _find_function(class_def: ast.ClassDef, function_name: str) -> ast.FunctionDef:
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"function {function_name} not found in {class_def.name}")


def _iter_calls(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            yield child


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword_value(call: ast.Call, keyword: str):
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


class TestQwen2TpaModelSurface(unittest.TestCase):
    def test_imports_attention_tp_helpers(self):
        module_ast = _load_module_ast()
        dp_imports = [
            node
            for node in module_ast.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "sglang.srt.layers.dp_attention"
        ]
        self.assertTrue(dp_imports, "expected dp_attention import in qwen2.py")

        imported_names = {alias.name for node in dp_imports for alias in node.names}
        self.assertIn("get_attention_tp_rank", imported_names)
        self.assertIn("get_attention_tp_size", imported_names)

    def test_attention_shards_qkv_and_o_proj_with_attention_tp(self):
        module_ast = _load_module_ast()
        attention_init = _find_function(
            _find_class(module_ast, "Qwen2Attention"), "__init__"
        )

        qkv_calls = [
            call for call in _iter_calls(attention_init) if _call_name(call) == "QKVParallelLinear"
        ]
        self.assertEqual(len(qkv_calls), 1)
        qkv_call = qkv_calls[0]
        self.assertEqual(
            getattr(_keyword_value(qkv_call, "tp_rank"), "id", None), "attn_tp_rank"
        )
        self.assertEqual(
            getattr(_keyword_value(qkv_call, "tp_size"), "id", None), "attn_tp_size"
        )

        row_calls = [
            call for call in _iter_calls(attention_init) if _call_name(call) == "RowParallelLinear"
        ]
        self.assertEqual(len(row_calls), 1)
        row_call = row_calls[0]
        self.assertEqual(
            getattr(_keyword_value(row_call, "tp_rank"), "id", None), "attn_tp_rank"
        )
        self.assertEqual(
            getattr(_keyword_value(row_call, "tp_size"), "id", None), "attn_tp_size"
        )
        self.assertIsInstance(_keyword_value(row_call, "reduce_results"), ast.Constant)
        self.assertFalse(_keyword_value(row_call, "reduce_results").value)

    def test_decoder_layer_uses_layer_communicator_flow(self):
        module_ast = _load_module_ast()
        decoder_class = _find_class(module_ast, "Qwen2DecoderLayer")
        decoder_init = _find_function(decoder_class, "__init__")
        decoder_forward = _find_function(decoder_class, "forward")

        communicator_calls = [
            call for call in _iter_calls(decoder_init) if _call_name(call) == "LayerCommunicator"
        ]
        self.assertEqual(len(communicator_calls), 1)

        forward_call_names = {_call_name(call) for call in _iter_calls(decoder_forward)}
        self.assertIn("prepare_attn", forward_call_names)
        self.assertIn("prepare_mlp", forward_call_names)
        self.assertIn("postprocess_layer", forward_call_names)


if __name__ == "__main__":
    unittest.main()
