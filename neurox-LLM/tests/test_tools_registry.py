"""core.tools: Tool schema + ToolRegistry.call error handling."""
from core.tools import Tool, ToolOutput, ToolRegistry


def _add(a: int, b: int) -> int:
    return a + b


def _boom():
    raise RuntimeError("kaput")


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("add", "Add two ints", {"type": "object", "properties": {"a": {"type": "integer"},
                                                                                "b": {"type": "integer"}},
                                              "required": ["a", "b"]}, _add))
    reg.register(Tool("boom", "Always fails", {}, _boom))
    reg.register(Tool("img", "Returns an image", {}, lambda: ToolOutput("shot", ["aGVsbG8="])))
    reg.register(Tool("none", "Returns None", {}, lambda: None))
    reg.register(Tool("rm", "dangerous", {}, lambda: "x", dangerous=True, category="files"))
    return reg


def test_schema_is_openai_compatible():
    reg = make_registry()
    s = reg.get("add").schema()
    assert s["type"] == "function"
    assert s["function"]["name"] == "add"
    assert s["function"]["parameters"]["required"] == ["a", "b"]
    # empty parameters are replaced by a valid empty object schema
    assert reg.get("boom").schema()["function"]["parameters"] == {"type": "object", "properties": {}}
    assert len(reg.schemas()) == len(reg) == 5


def test_call_with_dict_and_json_string():
    reg = make_registry()
    assert reg.call("add", {"a": 2, "b": 3}).text == "5"
    assert reg.call("add", '{"a": 4, "b": 5}').text == "9"


def test_empty_string_and_none_args():
    reg = make_registry()
    assert reg.call("none", "").text == ""
    assert reg.call("none", None).text == ""
    assert reg.call("none", "   ").text == ""


def test_bad_json_args_returns_error_text():
    out = make_registry().call("add", "{a: 1,")
    assert isinstance(out, ToolOutput)
    assert out.text.startswith("Error") and "JSON" in out.text


def test_unknown_tool_lists_available():
    out = make_registry().call("nope", {})
    assert "unknown tool" in out.text and "add" in out.text


def test_bad_arguments_are_captured():
    out = make_registry().call("add", {"a": 1})
    assert out.text.startswith("Error: bad arguments")
    out = make_registry().call("add", {"a": 1, "b": 2, "c": 3})
    assert out.text.startswith("Error: bad arguments")


def test_tool_exception_is_captured_not_raised():
    out = make_registry().call("boom", {})
    assert "RuntimeError" in out.text and "kaput" in out.text


def test_tool_output_with_images_passes_through():
    out = make_registry().call("img", {})
    assert out.text == "shot" and out.images == ["aGVsbG8="]


def test_register_unregister_filter_contains():
    reg = make_registry()
    assert "add" in reg and "zzz" not in reg
    reg.unregister("add")
    reg.unregister("does-not-exist")  # no error
    assert "add" not in reg
    sub = reg.filtered(["boom", "rm", "unknown"])
    assert sorted(sub.names()) == ["boom", "rm"]
    assert sorted(reg.filtered(None).names()) == sorted(reg.names())
    assert sub.get("rm").dangerous is True and sub.get("rm").category == "files"
    assert reg.get("missing") is None
