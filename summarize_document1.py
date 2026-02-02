import torch
import re
from transformers import BartTokenizer, BartForConditionalGeneration

# =============================
# CONFIG
# =============================
MODEL_PATH = "./bart_dailymail_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# LOAD MODEL (ONE TIME)
# =============================
print(f"Loading model on {DEVICE}...")
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("✓ Model ready\n")

# =============================
# CORE SUMMARIZATION ENGINE
# =============================
class AbstractiveSummarizer:
    """
    Enhanced abstractive summarizer with techniques to force true abstraction.
    """
    
    def __init__(self):
        self.tokenizer = tokenizer
        self.model = model
        self.device = DEVICE
    
    def _preprocess(self, text):
        """Clean input text."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_key_info(self, text):
        """
        Extract key information structure from text.
        This helps guide abstractive generation.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Categorize sentences by type
        intro_sentences = []
        benefit_sentences = []
        challenge_sentences = []
        conclusion_sentences = []
        other_sentences = []
        
        # Keywords for categorization
        intro_keywords = ['introduction', 'transforming', 'changing', 'rapidly', 'increasingly', 'global', 'transition']
        benefit_keywords = ['benefit', 'advantage', 'improve', 'enable', 'help', 'allow', 'provide', 'automation', 'accessible']
        challenge_keywords = ['challenge', 'concern', 'problem', 'issue', 'however', 'despite', 'but', 'difficulty', 'obstacle']
        conclusion_keywords = ['conclusion', 'therefore', 'thus', 'overall', 'essential', 'must', 'require', 'need']
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            
            # First sentence usually intro
            if i == 0 or any(kw in sent_lower for kw in intro_keywords):
                intro_sentences.append(sent)
            elif any(kw in sent_lower for kw in conclusion_keywords) or i >= len(sentences) - 2:
                conclusion_sentences.append(sent)
            elif any(kw in sent_lower for kw in challenge_keywords):
                challenge_sentences.append(sent)
            elif any(kw in sent_lower for kw in benefit_keywords):
                benefit_sentences.append(sent)
            else:
                other_sentences.append(sent)
        
        return {
            'intro': intro_sentences,
            'benefits': benefit_sentences,
            'challenges': challenge_sentences,
            'conclusion': conclusion_sentences,
            'other': other_sentences
        }
    
    def _create_guided_input(self, text, info_structure):
        """
        Create input that guides the model to cover all sections.
        This forces better coverage and abstraction.
        """
        # Build a structured prompt that emphasizes all parts
        parts = []
        
        # Intro (always include)
        if info_structure['intro']:
            parts.extend(info_structure['intro'][:2])
        
        # Benefits (critical - often skipped)
        if info_structure['benefits']:
            parts.extend(info_structure['benefits'])
        
        # Other important content
        if info_structure['other']:
            parts.extend(info_structure['other'][:3])
        
        # Challenges (usually captured)
        if info_structure['challenges']:
            parts.extend(info_structure['challenges'])
        
        # Conclusion (usually captured)
        if info_structure['conclusion']:
            parts.extend(info_structure['conclusion'])
        
        # Join with emphasis on transitions
        guided_text = ' '.join(parts)
        
        return guided_text
    
    def _get_length_params(self, word_count, size_option):
        """
        Calculate summary length based on size option.
        More aggressive lengths to force abstraction.
        """
        ratios = {
            "short": (0.20, 0.28),
            "medium": (0.32, 0.42),
            "long": (0.45, 0.60)
        }
        
        min_ratio, max_ratio = ratios.get(size_option, ratios["medium"])
        
        min_words = int(word_count * min_ratio)
        max_words = int(word_count * max_ratio)
        
        # Convert to tokens
        min_tokens = int(min_words * 1.3)
        max_tokens = int(max_words * 1.3)
        
        # Generous bounds to allow full coverage
        min_tokens = max(60, min(min_tokens, 350))
        max_tokens = max(120, min(max_tokens, 800))
        
        if max_tokens <= min_tokens:
            max_tokens = min_tokens + 60
        
        return max_tokens, min_tokens
    
    def _get_generation_params(self, context_level):
        """
        Generation parameters optimized for abstraction.
        """
        params = {
            "minimal": {
                "num_beams": 6,
                "length_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
                "early_stopping": True,
                "diversity_penalty": 0.3,  # Encourage varied phrasing
            },
            "balanced": {
                "num_beams": 8,
                "length_penalty": 1.3,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.15,
                "early_stopping": True,
                "diversity_penalty": 0.5,
            },
            "detailed": {
                "num_beams": 10,
                "length_penalty": 1.5,
                "no_repeat_ngram_size": 4,
                "repetition_penalty": 1.1,
                "early_stopping": True,
                "diversity_penalty": 0.7,
            }
        }
        
        return params.get(context_level, params["balanced"])
    
    def _generate(self, text, max_len, min_len, gen_params):
        """Core generation with abstraction focus."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                **gen_params
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_output(summary)
    
    def _generate_multi_candidate(self, text, max_len, min_len, gen_params, num_candidates=3):
        """
        Generate multiple candidates and select best.
        This increases chances of getting good abstraction.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False
        ).to(self.device)
        
        with torch.no_grad():
            # Generate multiple candidates
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                num_return_sequences=num_candidates,
                **gen_params
            )
        
        # Decode all candidates
        candidates = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        # Select best candidate (longest, most complete)
        best_candidate = max(candidates, key=lambda x: len(x.split()))
        
        return self._clean_output(best_candidate)
    
    def _clean_output(self, text):
        """Remove incomplete sentences and clean formatting."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        complete = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10 and sent[-1] in '.!?':
                complete.append(sent)
        
        if not complete:
            return text.strip()
        
        result = ' '.join(complete)
        result = re.sub(r'\s+([.,;:!?])', r'\1', result)
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def _split_into_segments(self, text, max_tokens=850):
        """Split long text into sentence-boundary segments."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        segments = []
        current = []
        current_len = 0
        
        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent, add_special_tokens=False))
            
            if current_len + sent_tokens > max_tokens and current:
                segments.append(' '.join(current))
                current = [sent]
                current_len = sent_tokens
            else:
                current.append(sent)
                current_len += sent_tokens
        
        if current:
            segments.append(' '.join(current))
        
        return segments
    
    def summarize(self, text, size="medium", context="balanced", use_multi_candidate=True):
        """
        Main summarization function with enhanced abstraction.
        
        Parameters:
        -----------
        text : str
            Input text to summarize
        
        size : str, optional (default="medium")
            Summary length: "short", "medium", or "long"
        
        context : str, optional (default="balanced")
            Context level: "minimal", "balanced", or "detailed"
        
        use_multi_candidate : bool, optional (default=True)
            Generate multiple candidates and select best
        
        Returns:
        --------
        str
            Generated summary
        """
        
        # Validate inputs
        if not text or len(text.strip()) < 50:
            return "Error: Input text too short (minimum 50 characters required)"
        
        size = size.lower()
        context = context.lower()
        
        if size not in ["short", "medium", "long"]:
            size = "medium"
        
        if context not in ["minimal", "balanced", "detailed"]:
            context = "balanced"
        
        # Preprocess
        text = self._preprocess(text)
        word_count = len(text.split())
        
        # Extract structure to guide generation
        info_structure = self._extract_key_info(text)
        
        # Create guided input for better coverage
        guided_text = self._create_guided_input(text, info_structure)
        
        # Get parameters
        max_len, min_len = self._get_length_params(word_count, size)
        gen_params = self._get_generation_params(context)
        
        # Choose generation method
        generate_fn = self._generate_multi_candidate if use_multi_candidate else self._generate
        
        # Short documents: direct summarization
        if word_count <= 600:
            if use_multi_candidate:
                summary = generate_fn(guided_text, max_len, min_len, gen_params, num_candidates=3)
            else:
                summary = generate_fn(guided_text, max_len, min_len, gen_params)
            return summary
        
        # Long documents: hierarchical summarization
        segments = self._split_into_segments(text)
        
        if len(segments) == 1:
            if use_multi_candidate:
                summary = generate_fn(guided_text, max_len, min_len, gen_params, num_candidates=3)
            else:
                summary = generate_fn(guided_text, max_len, min_len, gen_params)
            return summary
        
        # Multi-segment processing
        segment_summaries = []
        
        for segment in segments:
            seg_words = len(segment.split())
            
            # Extract structure for each segment
            seg_structure = self._extract_key_info(segment)
            seg_guided = self._create_guided_input(segment, seg_structure)
            
            # More aggressive compression for segments
            seg_max = int(seg_words * 0.45 * 1.3)
            seg_min = int(seg_words * 0.35 * 1.3)
            
            seg_max = max(80, min(seg_max, 300))
            seg_min = max(40, min(seg_min, 180))
            
            seg_summary = self._generate(seg_guided, seg_max, seg_min, gen_params)
            segment_summaries.append(seg_summary)
        
        # Combine and create final summary
        combined = ' '.join(segment_summaries)
        
        # Final pass with multi-candidate
        if use_multi_candidate:
            final_summary = self._generate_multi_candidate(combined, max_len, min_len, gen_params, num_candidates=3)
        else:
            final_summary = self._generate(combined, max_len, min_len, gen_params)
        
        return final_summary

# =============================
# INITIALIZE SUMMARIZER
# =============================
summarizer = AbstractiveSummarizer()

# =============================
# APP INTERFACE FUNCTION
# =============================
def generate_summary(text, summary_size="medium", context_level="balanced", quality_mode="best"):
    """
    Main function for your app.
    
    Parameters:
    -----------
    text : str
        Input document text
    
    summary_size : str
        Options: "short", "medium", "long"
        - short: ~20-28% of original
        - medium: ~32-42% of original (recommended)
        - long: ~45-60% of original
    
    context_level : str
        Options: "minimal", "balanced", "detailed"
        - minimal: Key facts, concise
        - balanced: Good coverage (recommended)
        - detailed: Maximum detail and nuance
    
    quality_mode : str
        Options: "fast", "best"
        - fast: Single candidate (faster)
        - best: Multiple candidates, select best (recommended)
    
    Returns:
    --------
    dict
        {
            'summary': str,
            'original_words': int,
            'summary_words': int,
            'compression_ratio': float
        }
    """
    
    use_multi = (quality_mode == "best")
    
    # Generate summary
    summary = summarizer.summarize(
        text, 
        size=summary_size, 
        context=context_level,
        use_multi_candidate=use_multi
    )
    
    # Calculate metrics
    original_words = len(text.split())
    summary_words = len(summary.split())
    compression = round((summary_words / original_words) * 100, 1) if original_words > 0 else 0
    
    return {
        'summary': summary,
        'original_words': original_words,
        'summary_words': summary_words,
        'compression_ratio': compression
    }

# =============================
# USAGE EXAMPLES
# =============================
if __name__ == "__main__":
    
    # Read input
    with open("news_article.txt", "r", encoding="utf-8") as f:
        article = f.read()
    
    print("="*70)
    print("ENHANCED ABSTRACTIVE SUMMARIZATION")
    print("="*70)
    print(f"\nOriginal text: {len(article.split())} words\n")
    
    # ===========================
    # RECOMMENDED: Long + Detailed + Best Quality
    # ===========================
    print("\n" + "="*70)
    print("RECOMMENDED: Long + Detailed + Best Quality")
    print("="*70)
    
    result = generate_summary(
        article, 
        summary_size="long", 
        context_level="detailed",
        quality_mode="best"
    )
    
    print(f"\nSummary ({result['summary_words']} words, {result['compression_ratio']}% of original):")
    print("-" * 70)
    print(result['summary'])
    
    # ===========================
    # FAST MODE: Medium + Balanced
    # ===========================
    print("\n\n" + "="*70)
    print("FAST MODE: Medium + Balanced")
    print("="*70)
    
    result = generate_summary(
        article, 
        summary_size="medium", 
        context_level="balanced",
        quality_mode="fast"
    )
    
    print(f"\nSummary ({result['summary_words']} words, {result['compression_ratio']}% of original):")
    print("-" * 70)
    print(result['summary'])
    
    # ===========================
    # BRIEF: Short + Minimal + Best
    # ===========================
    print("\n\n" + "="*70)
    print("BRIEF: Short + Minimal + Best Quality")
    print("="*70)
    
    result = generate_summary(
        article, 
        summary_size="short", 
        context_level="minimal",
        quality_mode="best"
    )
    
    print(f"\nSummary ({result['summary_words']} words, {result['compression_ratio']}% of original):")
    print("-" * 70)
    print(result['summary'])
    
    print("\n" + "="*70)
    print("DONE - Ready for production!")
    print("="*70)