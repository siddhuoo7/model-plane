"""conftest.py — shared pytest fixtures."""
import pytest


@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python function to reverse a linked list."},
    ]


@pytest.fixture
def sample_request(sample_messages):
    return {
        "model": "gpt-4o",
        "messages": sample_messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }
