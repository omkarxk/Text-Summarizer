"""
Backend Package
Professional Text Summarization Engine

Modules:
- chunking: Text splitting and chunking
- processing: Model loading and summarization
- io: File handling and text extraction
"""

from .hybrid_summarizer import HybridSummarizer
from .chunking import TextChunker
from .processing import ModelLoader, Summarizer
from .io import FileHandler

__all__ = [
    'HybridSummarizer',
    'TextChunker',
    'ModelLoader',
    'Summarizer',
    'FileHandler'
]
