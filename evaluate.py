"""
Summarization Evaluation Script
Calculates ROUGE, BLEU, and other metrics for summarization quality

Usage:
    python evaluate.py --input "Your text" --reference "Reference summary" --generated "Generated summary"
    
Or run interactively to evaluate your summaries.
"""

import argparse
import sys

# Check for required packages
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


def calculate_rouge(reference: str, generated: str) -> dict:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        reference: Reference/gold summary
        generated: Generated summary
        
    Returns:
        Dictionary with ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        return {"error": "rouge_score not installed. Run: pip install rouge-score"}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    
    return {
        'ROUGE-1': {
            'precision': round(scores['rouge1'].precision, 4),
            'recall': round(scores['rouge1'].recall, 4),
            'f1': round(scores['rouge1'].fmeasure, 4)
        },
        'ROUGE-2': {
            'precision': round(scores['rouge2'].precision, 4),
            'recall': round(scores['rouge2'].recall, 4),
            'f1': round(scores['rouge2'].fmeasure, 4)
        },
        'ROUGE-L': {
            'precision': round(scores['rougeL'].precision, 4),
            'recall': round(scores['rougeL'].recall, 4),
            'f1': round(scores['rougeL'].fmeasure, 4)
        }
    }


def calculate_bleu(reference: str, generated: str) -> dict:
    """
    Calculate BLEU score.
    
    Args:
        reference: Reference/gold summary
        generated: Generated summary
        
    Returns:
        Dictionary with BLEU score
    """
    if not BLEU_AVAILABLE:
        return {"error": "nltk not installed. Run: pip install nltk"}
    
    # Tokenize
    reference_tokens = reference.lower().split()
    generated_tokens = generated.lower().split()
    
    # Calculate BLEU with smoothing
    smoothing = SmoothingFunction().method1
    
    bleu_1 = sentence_bleu([reference_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return {
        'BLEU-1': round(bleu_1, 4),
        'BLEU-2': round(bleu_2, 4),
        'BLEU-4': round(bleu_4, 4)
    }


def calculate_compression(original: str, summary: str) -> dict:
    """
    Calculate compression metrics.
    
    Args:
        original: Original text
        summary: Generated summary
        
    Returns:
        Dictionary with compression metrics
    """
    original_words = len(original.split())
    summary_words = len(summary.split())
    original_chars = len(original)
    summary_chars = len(summary)
    
    compression_ratio = original_words / max(summary_words, 1)
    reduction_pct = (1 - summary_words / max(original_words, 1)) * 100
    
    return {
        'original_words': original_words,
        'summary_words': summary_words,
        'compression_ratio': round(compression_ratio, 2),
        'reduction_percentage': round(reduction_pct, 1),
        'original_chars': original_chars,
        'summary_chars': summary_chars
    }


def evaluate_summary(original: str, reference: str, generated: str) -> dict:
    """
    Run all evaluation metrics.
    
    Args:
        original: Original text
        reference: Reference/gold summary (can be None)
        generated: Generated summary
        
    Returns:
        Dictionary with all metrics
    """
    results = {
        'compression': calculate_compression(original, generated)
    }
    
    if reference:
        results['rouge'] = calculate_rouge(reference, generated)
        results['bleu'] = calculate_bleu(reference, generated)
    
    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("  SUMMARIZATION EVALUATION RESULTS")
    print("="*60)
    
    # Compression metrics
    comp = results.get('compression', {})
    print("\n📊 COMPRESSION METRICS:")
    print(f"   Original:    {comp.get('original_words', 'N/A')} words")
    print(f"   Summary:     {comp.get('summary_words', 'N/A')} words")
    print(f"   Compression: {comp.get('compression_ratio', 'N/A')}x")
    print(f"   Reduction:   {comp.get('reduction_percentage', 'N/A')}%")
    
    # ROUGE scores
    if 'rouge' in results and 'error' not in results['rouge']:
        print("\n📈 ROUGE SCORES (higher is better, max 1.0):")
        for metric, scores in results['rouge'].items():
            if isinstance(scores, dict):
                print(f"   {metric}: F1={scores['f1']:.4f} (P={scores['precision']:.4f}, R={scores['recall']:.4f})")
    elif 'rouge' in results and 'error' in results['rouge']:
        print(f"\n⚠️ ROUGE: {results['rouge']['error']}")
    
    # BLEU scores
    if 'bleu' in results and 'error' not in results['bleu']:
        print("\n📈 BLEU SCORES (higher is better, max 1.0):")
        for metric, score in results['bleu'].items():
            print(f"   {metric}: {score:.4f}")
    elif 'bleu' in results and 'error' in results['bleu']:
        print(f"\n⚠️ BLEU: {results['bleu']['error']}")
    
    print("\n" + "="*60)
    
    # Interpretation guide
    print("\n💡 INTERPRETATION GUIDE:")
    print("   ROUGE-1 F1 > 0.40: Good unigram overlap")
    print("   ROUGE-2 F1 > 0.20: Good bigram overlap")
    print("   ROUGE-L F1 > 0.35: Good longest common subsequence")
    print("   BLEU-4 > 0.30: Good n-gram precision")
    print()


def interactive_mode():
    """Run evaluation interactively."""
    print("\n" + "="*60)
    print("  SUMMARIZATION EVALUATION - Interactive Mode")
    print("="*60)
    
    print("\nPaste your ORIGINAL TEXT (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    original = " ".join(lines)
    
    print("\nPaste your REFERENCE SUMMARY (or press Enter to skip):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    reference = " ".join(lines) if lines else None
    
    print("\nPaste your GENERATED SUMMARY (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    generated = " ".join(lines)
    
    if not original or not generated:
        print("❌ Error: Both original text and generated summary are required.")
        return
    
    results = evaluate_summary(original, reference, generated)
    print_results(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate summarization quality')
    parser.add_argument('--original', '-o', type=str, help='Original text')
    parser.add_argument('--reference', '-r', type=str, help='Reference summary (optional)')
    parser.add_argument('--generated', '-g', type=str, help='Generated summary')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.original and not args.generated):
        interactive_mode()
    else:
        if not args.original or not args.generated:
            print("❌ Error: Both --original and --generated are required.")
            print("   Or use --interactive for interactive mode.")
            sys.exit(1)
        
        results = evaluate_summary(args.original, args.reference, args.generated)
        print_results(results)


if __name__ == "__main__":
    main()
