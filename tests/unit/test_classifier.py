"""Unit tests for the task classifier."""
from model_plane.classifier.classifier import classify
from model_plane.classifier.features import extract_features
from model_plane.classifier.taxonomy import ComplexityTier, TaskType


def _req(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_code_debugging_classification():
    result = classify(extract_features(_req(
        "I have a Python traceback: AttributeError: 'NoneType' object. Why does this fail?"
    )))
    assert result.task_type == TaskType.CODE_DEBUGGING
    assert result.confidence >= 0.7


def test_summarization_classification():
    result = classify(extract_features(_req("Please summarize the following document into key points.")))
    assert result.task_type == TaskType.SUMMARIZATION


def test_translation_classification():
    result = classify(extract_features(_req("Translate the following text into French: Hello, world.")))
    assert result.task_type == TaskType.TRANSLATION
    assert result.confidence >= 0.8


def test_simple_qa_classification():
    result = classify(extract_features(_req("What is the capital of France?")))
    assert result.task_type in (TaskType.SIMPLE_QA, TaskType.UNKNOWN)


def test_code_generation_classification():
    result = classify(extract_features(_req(
        "Write a Python function that implements binary search on a sorted list."
    )))
    assert result.task_type == TaskType.CODE_GENERATION
    assert result.complexity_tier in (ComplexityTier.MEDIUM, ComplexityTier.COMPLEX)


def test_math_reasoning():
    result = classify(extract_features(_req(
        "Calculate the eigenvalues of this 3x3 matrix and derive the characteristic polynomial."
    )))
    assert result.task_type == TaskType.MATHEMATICAL_REASONING
