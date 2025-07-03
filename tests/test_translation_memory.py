from translation_rag.translation_memory import (
    TranslationMemory,
    load_fake_memory,
    memory_to_documents,
)


def test_basic_translation():
    tm = TranslationMemory()
    tm.add_entries(load_fake_memory())
    result = tm.translate_text("Hola, como estas?", "es", "en")
    assert "Hello" in result


def test_memory_to_documents():
    data = load_fake_memory()
    texts, metas = memory_to_documents(data)
    assert len(texts) == len(metas) == len(data)
