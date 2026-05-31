"""Configuration for the novels2llm pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Pipeline configuration."""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    INPUT_DIR = DATA_DIR / "jia_ting_luan_lun"
    OUTPUT_DIR = DATA_DIR / "output"
    CACHE_DIR = OUTPUT_DIR / "cache"
    PROMPTS_DIR = PROJECT_ROOT / "prompts"

    # Anthropic API (supports DeepSeek and other compatible providers)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    ANTHROPIC_MAX_TOKENS = 4096

    # Chunking
    TARGET_CHUNK_SIZE_CHARS = 6000
    MAX_CHUNK_SIZE_CHARS = 8000
    CHUNK_OVERLAP_CHARS = 200

    # NLP
    NLP_ENABLE_HANLP = True  # HanLP NER (may be slow on first run)
    NLP_ENABLE_JIEBA = True  # jieba segmentation + POS tagging

    # Pipeline
    NOVEL_LIMIT = None  # None = process all, or set to N for testing
    CACHE_ENABLED = True  # Cache chunk-level extraction results

    # Chapter detection regex: matches "第N章", "第一章", etc.
    CHAPTER_PATTERNS = [
        r'第[\d一二三四五六七八九十百千]+章',
        r'第\s*\d+\s*章',
    ]

    # Dialogue patterns
    DIALOGUE_QUOTES = ['\u201c', '\u201d']  # Chinese curly quotes ""

    @classmethod
    def get_prompt_path(cls, name: str) -> Path:
        return cls.PROMPTS_DIR / f"{name}.txt"

    @classmethod
    def ensure_dirs(cls) -> None:
        """Ensure all output directories exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
