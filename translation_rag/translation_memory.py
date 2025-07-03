import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterable


def _tokens(text: str) -> set[str]:
    """Convert text to a set of lowercase word tokens."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass
class MemoryEntry:
    source_lang: str
    target_lang: str
    source_sentence: str
    target_sentence: str
    tokens: set[str]


class TranslationMemory:
    """Simple translation memory using Jaccard similarity."""

    def __init__(self):
        self.entries: List[MemoryEntry] = []

    def add_entry(self, source_lang: str, target_lang: str, source_sentence: str, target_sentence: str) -> None:
        entry = MemoryEntry(
            source_lang=source_lang,
            target_lang=target_lang,
            source_sentence=source_sentence,
            target_sentence=target_sentence,
            tokens=_tokens(source_sentence),
        )
        self.entries.append(entry)

    def add_entries(self, data: Iterable[dict]) -> None:
        for item in data:
            self.add_entry(
                item["source_lang"],
                item["target_lang"],
                item["source_sentence"],
                item["target_sentence"],
            )

    def retrieve(self, sentence: str, source_lang: str, target_lang: str, k: int = 2) -> List[MemoryEntry]:
        query_tokens = _tokens(sentence)
        scored = []
        for entry in self.entries:
            if entry.source_lang == source_lang and entry.target_lang == target_lang:
                sim = _jaccard(query_tokens, entry.tokens)
                scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def translate_sentence(self, sentence: str, source_lang: str, target_lang: str) -> str:
        matches = self.retrieve(sentence, source_lang, target_lang, k=1)
        if matches and matches[0].tokens:
            return matches[0].target_sentence
        return f"[no translation for: {sentence.strip()}]"

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        translated = []
        for sent in sentences:
            if not sent:
                continue
            translated.append(self.translate_sentence(sent, source_lang, target_lang))
        return " ".join(translated)

# Seed data lives in the repository root
DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[1] / "seed_memory"


def load_fake_memory(directory: Path | str = DEFAULT_MEMORY_DIR) -> List[dict]:
    """Load seed translation memory entries from JSON files."""
    path = Path(directory)
    if not path.exists():
        return []

    entries: List[dict] = []
    for file in sorted(path.glob("*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    entries.extend(data)
        except Exception:
            continue
    return entries


def memory_to_documents(entries: Iterable[dict]) -> tuple[list[str], list[dict]]:
    """Convert memory entries to texts and metadatas for the RAG pipeline."""
    texts: list[str] = []
    metadatas: list[dict] = []
    for item in entries:
        texts.append(f"{item['source_sentence']} -> {item['target_sentence']}")
        metadatas.append({
            "source_lang": item["source_lang"],
            "target_lang": item["target_lang"],
        })
    return texts, metadatas
