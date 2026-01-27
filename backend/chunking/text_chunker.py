"""
Text Chunker Module
Splits large documents into manageable chunks for processing
"""

import re
from typing import List


class TextChunker:
    """
    Handles intelligent text chunking for large documents.
    
    Supports multiple chunking strategies:
    - Simple word-based splitting
    - Sentence-aware splitting
    - Paragraph-aware splitting
    """
    
    def __init__(self, max_words: int = 750, overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            max_words: Maximum words per chunk (default 750 for BART's 1024 token limit)
            overlap: Number of overlapping words between chunks
        """
        self.max_words = max_words
        self.overlap = overlap
    
    def chunk_by_words(self, text: str, max_words: int = None) -> List[str]:
        """
        Simple word-based chunking.
        
        Args:
            text: Input text to chunk
            max_words: Override default max_words
            
        Returns:
            List of text chunks
        """
        max_words = max_words or self.max_words
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_sentences(self, text: str, max_words: int = None) -> List[str]:
        """
        Sentence-aware chunking - doesn't break mid-sentence.
        
        Args:
            text: Input text to chunk
            max_words: Override default max_words
            
        Returns:
            List of text chunks
        """
        max_words = max_words or self.max_words
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > max_words and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str, max_words: int = None) -> List[str]:
        """
        Paragraph-aware chunking - preserves paragraph boundaries.
        
        Args:
            text: Input text to chunk
            max_words: Override default max_words
            
        Returns:
            List of text chunks
        """
        max_words = max_words or self.max_words
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_words = len(para.split())
            
            if current_word_count + para_words > max_words and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_word_count = para_words
            else:
                current_chunk.append(para)
                current_word_count += para_words
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def smart_chunk(self, text: str, max_words: int = None) -> List[str]:
        """
        Intelligent chunking - uses best strategy based on text structure.
        
        Args:
            text: Input text to chunk
            max_words: Override default max_words
            
        Returns:
            List of text chunks
        """
        max_words = max_words or self.max_words
        
        # Check if text has paragraph structure
        if '\n\n' in text:
            chunks = self.chunk_by_paragraphs(text, max_words)
        else:
            # Use sentence-aware chunking
            chunks = self.chunk_by_sentences(text, max_words)
        
        # If any chunk is still too large, fall back to word splitting
        final_chunks = []
        for chunk in chunks:
            if len(chunk.split()) > max_words * 1.2:  # 20% tolerance
                final_chunks.extend(self.chunk_by_words(chunk, max_words))
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def get_chunk_info(self, text: str) -> dict:
        """
        Get information about how text would be chunked.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with chunking statistics
        """
        word_count = len(text.split())
        chunks = self.smart_chunk(text)
        
        return {
            'total_words': word_count,
            'num_chunks': len(chunks),
            'avg_chunk_size': word_count // len(chunks) if chunks else 0,
            'needs_chunking': word_count > self.max_words
        }
