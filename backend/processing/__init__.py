"""
Processing Module
Handles model loading and text summarization
"""

from .model_loader import ModelLoader
from .summarizer import Summarizer

__all__ = ['ModelLoader', 'Summarizer']
