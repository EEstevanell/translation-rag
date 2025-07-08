from translation_rag.strategies import JaccardRAG, LevenshteinRAG
from translation_rag.translation_memory import (TranslationMemory,
                                                load_fake_memory,
                                                memory_to_documents)


def test_basic_translation():
    tm = TranslationMemory()
    tm.add_entries(load_fake_memory())
    result = tm.translate_text("Hola, como estas?", "es", "en")
    assert "Hello" in result


def test_levenshtein_retrieve():
    tm = TranslationMemory()
    tm.add_entries(load_fake_memory())
    results = tm.retrieve_levenshtein("Hola, como estas?", "es", "en", k=1)
    assert results
    assert results[0].target_sentence


def test_memory_to_documents():
    data = load_fake_memory()
    texts, metas = memory_to_documents(data)
    assert len(texts) == len(metas) == len(data)
    assert texts[0] == data[0]["source_sentence"]
    assert metas[0]["target_sentence"] == data[0]["target_sentence"]


def test_levenshtein_strategy():
    tm = TranslationMemory()
    tm.add_entries(load_fake_memory())
    strat = LevenshteinRAG(tm)
    context = strat.get_context("Hola, como estas?", "es", "en", k=1)
    assert context and "Hola" in context and "Hello" in context


def test_levenshtein_threshold():
    tm = TranslationMemory()
    tm.add_entry("en", "es", "hello", "hola")

    # High threshold should filter the result
    high = tm.retrieve_levenshtein("hey", "en", "es", k=1, threshold=0.9)
    assert high == []

    # Low threshold should return the entry
    low = tm.retrieve_levenshtein("hey", "en", "es", k=1, threshold=0.1)
    assert low


def test_jaccard_retrieve():
    tm = TranslationMemory()
    tm.add_entries(load_fake_memory())
    results = tm.retrieve_jaccard("Hola, como estas?", "es", "en", k=1)
    assert results
    assert results[0].target_sentence


def test_jaccard_strategy():
    tm = TranslationMemory()
    tm.add_entries(load_fake_memory())
    strat = JaccardRAG(tm)
    context = strat.get_context("Hola, como estas?", "es", "en", k=1)
    assert context and "Hola" in context and "Hello" in context
