"""Unit tests for response validator."""
from model_plane.validation.validator import validate_response


def test_valid_plain_text():
    result = validate_response("This is a helpful answer.")
    assert result.passed
    assert result.result_code == "ok"


def test_empty_response_fails():
    result = validate_response("")
    assert not result.passed
    assert result.result_code == "empty"
    assert result.should_retry


def test_none_response_fails():
    result = validate_response(None)
    assert not result.passed


def test_json_extraction_success():
    result = validate_response('{"name": "Alice", "age": 30}', expect_json=True)
    assert result.passed


def test_json_extraction_in_markdown():
    result = validate_response(
        'Here is the result:\n```json\n{"key": "value"}\n```',
        expect_json=True,
    )
    assert result.passed


def test_json_schema_validation_pass():
    result = validate_response(
        '{"name": "Alice", "score": 95}',
        expect_json=True,
        json_schema={"required": ["name", "score"]},
    )
    assert result.passed


def test_json_schema_validation_fail():
    result = validate_response(
        '{"name": "Alice"}',
        expect_json=True,
        json_schema={"required": ["name", "score"]},
    )
    assert not result.passed
    assert result.result_code == "schema_fail"
    assert result.should_escalate


def test_parse_error_escalates():
    result = validate_response("not json at all %%!", expect_json=True)
    assert not result.passed
    assert result.result_code == "parse_error"
    assert result.should_escalate


def test_tool_call_valid():
    result = validate_response('{"name": "get_weather", "arguments": {}}', expect_tool_call=True)
    assert result.passed


def test_tool_call_missing_fails():
    result = validate_response("I cannot call tools right now.", expect_tool_call=True)
    assert not result.passed
    assert result.result_code == "tool_call_fail"
