"""
PROFESSIONAL TEXT SUMMARIZER - Main Entry Point
Orchestrates all components for document summarization

Uses modular architecture:
- chunking/: Text splitting for large documents
- processing/: ML model loading and inference  
- io/: File input/output handling
"""

from .chunking import TextChunker
from .processing import ModelLoader, Summarizer


class HybridSummarizer:
    """
    Main Summarizer Class - Orchestrates all components
    
    Provides a clean interface for document summarization with:
    - Automatic chunking for large documents
    - GPU-accelerated inference
    - Configurable summary size and detail level
    """
    
    def __init__(self, model_name: str = "t5-base"):
        """
        Initialize all components.
        Uses Pegasus-xsum - state-of-the-art abstractive summarization!
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.is_t5_model = "t5" in model_name.lower()
        self.is_pegasus_model = "pegasus" in model_name.lower()
        
        # Initialize components
        self.chunker = TextChunker(max_words=750)
        self.model_loader = ModelLoader()
        
        # Load the appropriate model
        if self.is_pegasus_model:
            model, tokenizer = self.model_loader.load_pegasus(model_name)
        elif self.is_t5_model:
            model, tokenizer = self.model_loader.load_t5(model_name)
        else:
            model, tokenizer = self.model_loader.load_bart(model_name)
        
        # Initialize summarizer with loaded model
        self.summarizer = Summarizer(
            model=model,
            tokenizer=tokenizer,
            device=self.model_loader.device,
            is_t5_model=self.is_t5_model
        )
    
    def summarize_text(self, text: str, max_length: int = 150, 
                       min_length: int = 40) -> str:
        """
        Generate summary for a single text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary tokens
            min_length: Minimum summary tokens
            
        Returns:
            Generated summary
        """
        return self.summarizer.summarize_single(text, max_length, min_length)
    
    def summarize_batch(self, texts: list, max_length: int = 150,
                        min_length: int = 40) -> list:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum summary tokens
            min_length: Minimum summary tokens
            
        Returns:
            List of generated summaries
        """
        return self.summarizer.summarize_batch(texts, max_length, min_length)
    
    def chunk_text(self, text: str, max_words: int = 750) -> list:
        """
        Split text into chunks for processing.
        
        Args:
            text: Input text
            max_words: Maximum words per chunk
            
        Returns:
            List of text chunks
        """
        return self.chunker.smart_chunk(text, max_words)
    
    def abstractive_summary(self, text: str, max_length: int = 150,
                            min_length: int = 30) -> str:
        """Wrapper for compatibility."""
        return self.summarize_text(text, max_length, min_length)
    
    def hybrid_summarize(self, text: str, summary_size: str = "medium",
                         context_level: str = "balanced") -> dict:
        """
        MAIN METHOD: Professional summarization with size/context control
        
        Args:
            text: Document text to summarize
            summary_size: very_small, small, medium, large, very_large
            context_level: simple, balanced, detailed
            
        Returns:
            Dictionary with 'extracted' and 'final_summary'
        """
        print(f"📝 Received - Size: '{summary_size}', Context: '{context_level}'")
        
        # Get generation config from summarizer
        config = self.summarizer.get_generation_config(summary_size, context_level)
        max_len = config['max_length']
        min_len = config['min_length']
        chunk_len = config['chunk_length']
        
        print(f"⚙️ Config - Max: {max_len}, Min: {min_len} tokens")
        
        word_count = len(text.split())
        print(f"📄 Document: {word_count} words")
        
        if word_count > 800:
            # Large document - hierarchical summarization
            print("🔄 Processing large document...")
            
            # Use smart chunking
            chunks = self.chunker.smart_chunk(text, max_words=750)
            print(f"📊 Split into {len(chunks)} chunks")
            
            # Process in batches
            batch_size = 4
            all_summaries = []
            
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                print(f"⏳ Processing batch {batch_num}/{total_batches}...")
                
                batch_summaries = self.summarizer.summarize_batch(
                    batch,
                    max_length=chunk_len,
                    min_length=40
                )
                all_summaries.extend(batch_summaries)
            
            # Combine intermediate summaries
            combined = ' '.join(all_summaries)
            combined_words = len(combined.split())
            print(f"📝 Intermediate: {combined_words} words")
            
            # If still long, do another pass
            if combined_words > 800:
                print("🔄 Second pass...")
                chunks2 = self.chunker.smart_chunk(combined, max_words=750)
                summaries2 = []
                for chunk in chunks2:
                    s = self.summarizer.summarize_single(
                        chunk, max_length=250, min_length=60
                    )
                    summaries2.append(s)
                combined = ' '.join(summaries2)
            
            # Final summary
            print("✨ Creating final summary...")
            final_summary = self.summarizer.summarize_single(
                combined, max_length=max_len, min_length=min_len
            )
        else:
            # Small document - direct summarization
            print("✨ Generating summary...")
            final_summary = self.summarizer.summarize_single(
                text, max_length=max_len, min_length=min_len
            )
        
        word_count_out = len(final_summary.split())
        print(f"✅ Summary: {word_count_out} words")
        
        return {
            "extracted": "",
            "final_summary": final_summary
        }
    
    def get_status(self) -> dict:
        """
        Get system status and model info.
        
        Returns:
            Dictionary with system status
        """
        return {
            'model': self.model_name,
            'device': str(self.model_loader.device),
            'chunker_max_words': self.chunker.max_words,
            **self.model_loader.get_model_info()
        }
