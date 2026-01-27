"""
Summarizer Module
Core summarization logic using transformer models
"""

import torch
import re
from typing import List


class Summarizer:
    """
    Handles text summarization using transformer models.
    Supports single text and batch processing.
    """
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize the summarizer with a model.
        
        Args:
            model: Loaded transformer model
            tokenizer: Corresponding tokenizer
            device: torch device (cuda/mps/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def _ensure_complete_sentence(self, text: str) -> str:
        """
        Ensure the text ends with a complete sentence.
        Removes any trailing incomplete sentence.
        """
        text = text.strip()
        if not text:
            return text
        
        # If already ends with sentence-ending punctuation, return as-is
        if text[-1] in '.!?':
            return text
        
        # Find the last complete sentence
        # Look for the last occurrence of . ! or ? followed by space or end
        last_period = text.rfind('. ')
        last_exclaim = text.rfind('! ')
        last_question = text.rfind('? ')
        
        # Also check for sentence ending at the very end (before incomplete part)
        for i in range(len(text) - 1, -1, -1):
            if text[i] in '.!?':
                return text[:i + 1]
        
        # If no sentence ending found, add a period
        return text + '.'
    
    def summarize_single(self, text: str, max_length: int = 150, 
                         min_length: int = 40) -> str:
        """
        Generate summary for a single text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            
        Returns:
            Generated summary string (complete sentences only)
        """
        # Ensure valid lengths (fix for MPS overflow error)
        max_length = min(max(max_length, 50), 512)
        min_length = min(max(min_length, 10), max_length - 20)
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        # Move to device - ensure proper dtypes for MPS
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate summary with extra buffer for sentence completion
        try:
            with torch.no_grad():
                output = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=int(max_length + 20),  # Explicit int for MPS
                    min_length=int(min_length),
                    num_beams=4,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=False,
                )
        except RuntimeError as e:
            # Fallback to CPU if MPS fails
            if "integral" in str(e) or "mps" in str(e).lower():
                print("⚠️ MPS error, falling back to CPU for this operation")
                cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
                self.model.to("cpu")
                with torch.no_grad():
                    output = self.model.generate(
                        cpu_inputs["input_ids"],
                        attention_mask=cpu_inputs["attention_mask"],
                        max_length=int(max_length + 20),
                        min_length=int(min_length),
                        num_beams=4,
                        length_penalty=2.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        do_sample=False,
                    )
                self.model.to(self.device)
            else:
                raise e
        
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Ensure complete sentences
        summary = self._ensure_complete_sentence(summary)
        
        return summary
    
    def summarize_batch(self, texts: List[str], max_length: int = 150,
                        min_length: int = 40) -> List[str]:
        """
        Generate summaries for multiple texts in batch.
        More efficient than calling summarize_single multiple times.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            
        Returns:
            List of generated summaries (complete sentences only)
        """
        if not texts:
            return []
        
        # Ensure valid lengths (fix for MPS overflow error)
        max_length = min(max(max_length, 50), 512)
        min_length = min(max(min_length, 10), max_length - 20)
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=int(max_length + 20),  # Explicit int for MPS
                    min_length=int(min_length),
                    num_beams=3,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=False,
                )
        except RuntimeError as e:
            # Fallback to CPU if MPS fails
            if "integral" in str(e) or "mps" in str(e).lower():
                print("⚠️ MPS error in batch, falling back to CPU")
                cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
                self.model.to("cpu")
                with torch.no_grad():
                    outputs = self.model.generate(
                        cpu_inputs["input_ids"],
                        attention_mask=cpu_inputs["attention_mask"],
                        max_length=int(max_length + 20),
                        min_length=int(min_length),
                        num_beams=3,
                        length_penalty=2.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        do_sample=False,
                    )
                self.model.to(self.device)
            else:
                raise e
        
        summaries = []
        for output in outputs:
            summary = self.tokenizer.decode(output, skip_special_tokens=True)
            summary = self._ensure_complete_sentence(summary)
            summaries.append(summary)
        
        return summaries
    
    def get_generation_config(self, summary_size: str, 
                               context_level: str) -> dict:
        """
        Get generation parameters based on size and context settings.
        
        Args:
            summary_size: very_small, small, medium, large, very_large
            context_level: simple, balanced, detailed
            
        Returns:
            Dictionary with max_length, min_length, chunk_length
        """
        # Base configuration for each size
        size_config = {
            "very_small": (80, 30, 60),
            "small": (150, 60, 100),
            "medium": (250, 100, 150),
            "large": (400, 180, 250),
            "very_large": (600, 300, 350)
        }
        
        # Context level adjustments
        context_adjust = {
            "simple": -40,
            "balanced": 0,
            "detailed": 60
        }
        
        # Get base configuration
        config = size_config.get(summary_size, (250, 100, 150))
        max_len, min_len, chunk_len = config
        
        # Apply context adjustment
        adjustment = context_adjust.get(context_level, 0)
        max_len = min(800, max_len + adjustment)
        min_len = max(20, min_len + adjustment // 2)
        chunk_len = min(500, chunk_len + adjustment // 2)
        
        return {
            'max_length': max_len,
            'min_length': min_len,
            'chunk_length': chunk_len
        }
