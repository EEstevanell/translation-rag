import os
from utils import save_translation_data, load_translation_data, create_sample_translation_data


def test_save_translation_data_no_directory(tmp_path, monkeypatch):
    # change to temporary directory
    monkeypatch.chdir(tmp_path)
    file_name = "translation_data.json"
    data = create_sample_translation_data()
    assert save_translation_data(data, file_name)
    loaded = load_translation_data(file_name)
    assert loaded and loaded[0]["id"] == "greeting_1"

