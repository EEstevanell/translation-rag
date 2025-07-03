import os
from translation_rag.utils import (
    save_translation_data,
    load_translation_data,
    create_sample_translation_data,
    render_translation_prompt,
)


def test_save_translation_data_no_directory(tmp_path, monkeypatch):
    # change to temporary directory
    monkeypatch.chdir(tmp_path)
    file_name = "translation_data.json"
    data = create_sample_translation_data()
    assert save_translation_data(data, file_name)
    loaded = load_translation_data(file_name)
    assert loaded and loaded[0]["id"] == "greeting_1"


def test_render_translation_prompt_basic():
    prompt = render_translation_prompt("Hello", "en", "es")
    assert "en" in prompt and "es" in prompt

