"""
Model Loader Module
Handles loading and managing ML models with GPU acceleration
"""

import torch
import warnings
from typing import Tuple, Any

warnings.filterwarnings("ignore")


class ModelLoader:
    """
    Handles loading transformer models with automatic device detection.
    Supports CUDA, Apple Silicon MPS, and CPU fallback.
    """
    
    def __init__(self):
        """Initialize and detect best available device."""
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None
        self.model_name = None
    
    def _detect_device(self) -> torch.device:
        """
        Detect the best available device for inference.
        
        Returns:
            torch.device: Best available device (MPS, CUDA, or CPU)
        """
        if torch.backends.mps.is_available():
            print("🚀 Using Apple Silicon GPU (MPS) - Fast Mode!")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("🚀 Using NVIDIA GPU (CUDA) - Fast Mode!")
            return torch.device("cuda")
        else:
            print("⚠️ Using CPU (slower)")
            return torch.device("cpu")
    
    def load_bart(self, model_name: str = "sshleifer/distilbart-cnn-6-6") -> Tuple[Any, Any]:
        """
        Load BART model for summarization.
        Uses DistilBART for faster inference - 50% faster than BART-large!
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import BartForConditionalGeneration, BartTokenizer
        
        print(f"📦 Loading {model_name} (lightweight & fast)...")
        
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
    
    def load_t5(self, model_name: str = "t5-base") -> Tuple[Any, Any]:
        """
        Load T5 model for text-to-text tasks.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        print(f"📦 Loading {model_name}...")
        
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
    
    def load_pegasus(self, model_name: str = "google/pegasus-xsum") -> Tuple[Any, Any]:
        """
        Load Pegasus model for abstractive summarization.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        
        print(f"📦 Loading {model_name}...")
        
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_name': self.model_name,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }
