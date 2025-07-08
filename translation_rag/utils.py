"""Utility functions for the Translation RAG system."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from langdetect import LangDetectException, detect

# Jinja environment for rendering prompt templates
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)


def load_translation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load translation data from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Translation data file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return []


def save_translation_data(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Save translation data to a JSON file."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving translation data to {file_path}: {e}")
        return False


def create_sample_translation_data():
    """Create sample translation data for testing."""
    sample_data = [
        {
            "id": "greeting_1",
            "type": "greeting",
            "translations": {
                "en": "Hello, how are you?",
                "es": "Hola, ¿cómo estás?",
                "fr": "Bonjour, comment allez-vous?",
                "de": "Hallo, wie geht es dir?",
                "it": "Ciao, come stai?",
                "pt": "Olá, como você está?",
                "zh": "你好吗？",
            },
            "context": "Common greeting",
            "formality": "informal",
        },
        {
            "id": "courtesy_1",
            "type": "courtesy",
            "translations": {
                "en": "Thank you very much",
                "es": "Muchas gracias",
                "fr": "Merci beaucoup",
                "de": "Vielen Dank",
                "it": "Grazie mille",
                "pt": "Muito obrigado",
                "zh": "非常感谢",
            },
            "context": "Expressing gratitude",
            "formality": "neutral",
        },
        {
            "id": "question_1",
            "type": "question",
            "translations": {
                "en": "Where is the bathroom?",
                "es": "¿Dónde está el baño?",
                "fr": "Où sont les toilettes?",
                "de": "Wo ist die Toilette?",
                "it": "Dove si trova il bagno?",
                "pt": "Onde fica o banheiro?",
                "zh": "厕所在哪里？",
            },
            "context": "Asking for directions",
            "formality": "neutral",
        },
        {
            "id": "business_1",
            "type": "business",
            "translations": {
                "en": "I would like to schedule a meeting",
                "es": "Me gustaría programar una reunión",
                "fr": "J'aimerais programmer une réunion",
                "de": "Ich möchte ein Meeting planen",
                "it": "Vorrei programmare una riunione",
                "pt": "Gostaria de agendar uma reunião",
                "zh": "我想安排一个会议",
            },
            "context": "Business communication",
            "formality": "formal",
        },
        {
            "id": "travel_1",
            "type": "travel",
            "translations": {
                "en": "How much does this cost?",
                "es": "¿Cuánto cuesta esto?",
                "fr": "Combien ça coûte?",
                "de": "Wie viel kostet das?",
                "it": "Quanto costa questo?",
                "pt": "Quanto custa isso?",
                "zh": "这个多少钱？",
            },
            "context": "Shopping/pricing inquiry",
            "formality": "neutral",
        },
    ]

    return sample_data


def format_translation_examples(data: List[Dict[str, Any]]) -> List[str]:
    """Format translation data for use in RAG system."""
    formatted_examples = []

    for item in data:
        translations = item.get("translations", {})
        context = item.get("context", "General")
        formality = item.get("formality", "neutral")

        # Create a formatted string for each translation set
        example_text = f"Context: {context} (Formality: {formality})\n"

        for lang, text in translations.items():
            lang_names = {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "zh": "Chinese",
            }
            lang_name = lang_names.get(lang, lang.upper())
            example_text += f"{lang_name}: {text}\n"

        formatted_examples.append(example_text.strip())

    return formatted_examples


def setup_sample_data_file(file_path: str = "translation_data.json") -> bool:
    """Create a sample translation data file if it doesn't exist."""
    if not os.path.exists(file_path):
        sample_data = create_sample_translation_data()
        return save_translation_data(sample_data, file_path)
    return True


def get_supported_languages() -> Dict[str, str]:
    """Get a dictionary of supported language codes and their names."""
    return {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
    }


DEFAULT_SYSTEM_PROMPT_TEMPLATE = "system_prompt.jinja"


def render_system_prompt(
    source_lang: str,
    target_lang: str,
    template_str: Optional[str] = None,
) -> str:
    """Render the system prompt inserting the requested languages."""
    if template_str is None:
        template = _jinja_env.get_template(DEFAULT_SYSTEM_PROMPT_TEMPLATE)
    else:
        from jinja2 import Template

        template = Template(template_str)
    return template.render(source_lang=source_lang, target_lang=target_lang)


def render_translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    system_message: str,
    context: Optional[str] = None,
) -> str:
    """Render the translation prompt using the Jinja template."""
    template = _jinja_env.get_template("translation_prompt.jinja")
    return template.render(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        supported_languages=", ".join(get_supported_languages().values()),
        system_message=system_message,
        context=context,
    )


def detect_language(text: str) -> str:
    """Return ISO language code detected in the given text."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def clean_chroma_db(persist_directory: str = "./chroma_db") -> bool:
    """Clean the ChromaDB persistent directory."""
    try:
        import shutil

        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"✓ Cleaned ChromaDB directory: {persist_directory}")
            return True
        else:
            print(f"ChromaDB directory does not exist: {persist_directory}")
            return False
    except Exception as e:
        print(f"Error cleaning ChromaDB directory: {e}")
        return False


if __name__ == "__main__":
    # Create sample data file for testing
    setup_sample_data_file()
    print("Sample translation data file created: translation_data.json")
