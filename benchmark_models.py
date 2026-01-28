"""
Model Benchmark Script
Compare summarization quality across multiple models using ROUGE and BLEU metrics.

Models tested:
1. google/flan-t5-base
2. sshleifer/distilbart-cnn-6-6
3. facebook/bart-large-cnn
4. t5-base

Usage:
    python benchmark_models.py
"""

import time
import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer
)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings("ignore")

# Sample texts with reference summaries for evaluation
BENCHMARK_DATA = [
    {
        "id": 1,
        "name": "Technology Article",
        "text": """Artificial intelligence has transformed the way businesses operate across every industry. 
        Machine learning algorithms now power recommendation systems, fraud detection, and customer service chatbots. 
        Companies like Google, Amazon, and Microsoft have invested billions of dollars in AI research and development. 
        The healthcare sector has seen particularly impressive applications, with AI systems now capable of detecting 
        diseases from medical images with accuracy rivaling human doctors. However, concerns about job displacement, 
        privacy, and algorithmic bias continue to spark debate among policymakers and ethicists. The technology is 
        advancing rapidly, with new breakthroughs in natural language processing and computer vision announced almost 
        weekly. Experts predict that AI will add trillions of dollars to the global economy over the next decade, 
        while also fundamentally reshaping the nature of work and human-machine interaction.""",
        "reference": "Artificial intelligence is transforming businesses across industries through machine learning applications in recommendations, fraud detection, and healthcare. Major tech companies invest heavily in AI research. While AI promises economic benefits, concerns about jobs, privacy, and bias remain."
    },
    {
        "id": 2,
        "name": "Climate Change Report",
        "text": """Global temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times, 
        with the past decade being the warmest on record. Scientists attribute this warming primarily to human 
        activities, particularly the burning of fossil fuels which releases carbon dioxide and other greenhouse 
        gases into the atmosphere. The effects of climate change are already visible worldwide, including more 
        frequent extreme weather events, rising sea levels, and shifting ecosystems. Coral reefs are bleaching 
        at unprecedented rates, while Arctic ice coverage has declined by over 40% since satellite measurements 
        began. The Paris Agreement, signed by nearly 200 countries, aims to limit global warming to 1.5 degrees 
        Celsius, but current national commitments fall short of this goal. Renewable energy adoption is accelerating, 
        with solar and wind power costs dropping dramatically, yet fossil fuels still account for over 80% of 
        global energy consumption. Without significant policy changes and technological innovations, scientists 
        warn that the world faces severe consequences including widespread food and water shortages.""",
        "reference": "Global temperatures have risen 1.1°C since pre-industrial times due to fossil fuel emissions. Climate impacts include extreme weather, rising seas, and ecosystem changes. The Paris Agreement aims to limit warming to 1.5°C, but current commitments are insufficient despite growing renewable energy adoption."
    },
    {
        "id": 3,
        "name": "Economic Analysis",
        "text": """The global economy experienced unprecedented disruption during the pandemic years, with supply 
        chain bottlenecks and labor shortages driving inflation to levels not seen in decades. Central banks 
        responded by raising interest rates aggressively, aiming to cool overheated economies without triggering 
        recessions. The technology sector, which had seen explosive growth during lockdowns, faced significant 
        corrections as investors reassessed valuations in a higher interest rate environment. Meanwhile, the 
        energy sector benefited from elevated oil and gas prices, though this put additional pressure on consumers 
        and businesses already struggling with rising costs. Governments deployed massive fiscal stimulus packages, 
        adding trillions to national debts. As the immediate crisis faded, attention turned to structural challenges 
        including aging populations in developed nations, the transition to clean energy, and the reshaping of 
        global trade relationships amid geopolitical tensions. Economists remain divided on whether the world 
        economy will achieve a soft landing or face prolonged stagnation.""",
        "reference": "The pandemic caused global economic disruption with supply chain issues and inflation. Central banks raised rates while tech stocks fell and energy prices rose. Governments added to national debts with stimulus. Future challenges include aging populations, clean energy transition, and geopolitical tensions."
    }
]

# Models to benchmark
MODELS = [
    {
        "name": "FLAN-T5-base",
        "model_id": "google/flan-t5-base",
        "type": "t5",
        "size": "~990MB"
    },
    {
        "name": "DistilBART-CNN",
        "model_id": "sshleifer/distilbart-cnn-6-6",
        "type": "bart",
        "size": "~1.2GB"
    },
    {
        "name": "BART-large-CNN",
        "model_id": "facebook/bart-large-cnn",
        "type": "bart",
        "size": "~1.6GB"
    },
    {
        "name": "T5-base",
        "model_id": "t5-base",
        "type": "t5",
        "size": "~990MB"
    }
]


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_info, device):
    """Load a model and tokenizer."""
    model_id = model_info["model_id"]
    model_type = model_info["type"]
    
    try:
        if model_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(model_id)
            model = T5ForConditionalGeneration.from_pretrained(model_id)
        else:  # bart
            tokenizer = BartTokenizer.from_pretrained(model_id)
            model = BartForConditionalGeneration.from_pretrained(model_id)
        
        model = model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"  ⚠️ Failed to load {model_id}: {e}")
        return None, None


def generate_summary(model, tokenizer, text, model_type, device):
    """Generate a summary for the given text."""
    # Prepare input
    if model_type == "t5":
        input_text = f"summarize: {text}"
    else:
        input_text = text
    
    inputs = tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def calculate_rouge(reference, generated):
    """Calculate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure
    }


def calculate_bleu(reference, generated):
    """Calculate BLEU score."""
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)


def calculate_compression(original, summary):
    """Calculate compression ratio."""
    orig_words = len(original.split())
    summ_words = len(summary.split())
    return orig_words / max(summ_words, 1)


def benchmark_model(model_info, benchmark_data, device):
    """Benchmark a single model on all test data."""
    print(f"\n{'='*60}")
    print(f"  Loading {model_info['name']} ({model_info['size']})...")
    print(f"{'='*60}")
    
    start_load = time.time()
    model, tokenizer = load_model(model_info, device)
    load_time = time.time() - start_load
    
    if model is None:
        return None
    
    print(f"  ✅ Model loaded in {load_time:.1f}s")
    
    results = {
        "model_name": model_info["name"],
        "model_id": model_info["model_id"],
        "load_time": load_time,
        "samples": []
    }
    
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    total_bleu = 0
    total_time = 0
    
    for sample in benchmark_data:
        print(f"\n  📝 Processing: {sample['name']}...")
        
        start_gen = time.time()
        try:
            generated = generate_summary(model, tokenizer, sample["text"], model_info["type"], device)
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            # Fallback to CPU if MPS fails
            if device.type == "mps":
                print("    🔄 Retrying on CPU...")
                model = model.to("cpu")
                generated = generate_summary(model, tokenizer, sample["text"], model_info["type"], torch.device("cpu"))
                model = model.to(device)
            else:
                continue
        gen_time = time.time() - start_gen
        
        # Calculate metrics
        rouge_scores = calculate_rouge(sample["reference"], generated)
        bleu_score = calculate_bleu(sample["reference"], generated)
        compression = calculate_compression(sample["text"], generated)
        
        sample_result = {
            "name": sample["name"],
            "generated_summary": generated,
            "generation_time": gen_time,
            "rouge1_f1": rouge_scores["rouge1_f1"],
            "rouge2_f1": rouge_scores["rouge2_f1"],
            "rougeL_f1": rouge_scores["rougeL_f1"],
            "bleu4": bleu_score,
            "compression_ratio": compression
        }
        results["samples"].append(sample_result)
        
        total_rouge1 += rouge_scores["rouge1_f1"]
        total_rouge2 += rouge_scores["rouge2_f1"]
        total_rougeL += rouge_scores["rougeL_f1"]
        total_bleu += bleu_score
        total_time += gen_time
        
        print(f"    ⏱️ Time: {gen_time:.2f}s | ROUGE-1: {rouge_scores['rouge1_f1']:.3f} | ROUGE-2: {rouge_scores['rouge2_f1']:.3f} | BLEU-4: {bleu_score:.3f}")
    
    n = len(benchmark_data)
    results["avg_rouge1"] = total_rouge1 / n
    results["avg_rouge2"] = total_rouge2 / n
    results["avg_rougeL"] = total_rougeL / n
    results["avg_bleu4"] = total_bleu / n
    results["avg_gen_time"] = total_time / n
    
    # Clean up memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def print_comparison_table(all_results):
    """Print a comparison table of all models."""
    print("\n")
    print("=" * 100)
    print("  📊 MODEL COMPARISON - EVALUATION METRICS")
    print("=" * 100)
    
    # Header
    print(f"\n{'Model':<25} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12} {'BLEU-4':<12} {'Avg Time':<12}")
    print("-" * 85)
    
    # Sort by ROUGE-2 (most important metric)
    sorted_results = sorted(all_results, key=lambda x: x["avg_rouge2"], reverse=True)
    
    for r in sorted_results:
        print(f"{r['model_name']:<25} {r['avg_rouge1']:.4f}       {r['avg_rouge2']:.4f}       {r['avg_rougeL']:.4f}       {r['avg_bleu4']:.4f}       {r['avg_gen_time']:.2f}s")
    
    print("-" * 85)
    
    # Find best model
    best = sorted_results[0]
    print(f"\n🏆 BEST MODEL: {best['model_name']}")
    print(f"   ROUGE-1: {best['avg_rouge1']:.4f} | ROUGE-2: {best['avg_rouge2']:.4f} | ROUGE-L: {best['avg_rougeL']:.4f}")
    
    # Interpretation guide
    print("\n" + "=" * 100)
    print("  💡 METRIC INTERPRETATION")
    print("=" * 100)
    print("""
    ROUGE-1 (Unigram Overlap):
        > 0.45 = Excellent | 0.35-0.45 = Good | 0.25-0.35 = Fair | < 0.25 = Poor
    
    ROUGE-2 (Bigram Overlap) - Most Important:
        > 0.20 = Excellent | 0.15-0.20 = Good | 0.10-0.15 = Fair | < 0.10 = Poor
    
    ROUGE-L (Longest Common Subsequence):
        > 0.40 = Excellent | 0.30-0.40 = Good | 0.20-0.30 = Fair | < 0.20 = Poor
    
    BLEU-4 (N-gram Precision):
        > 0.30 = Excellent | 0.20-0.30 = Good | 0.10-0.20 = Fair | < 0.10 = Poor
    """)


def print_sample_summaries(all_results):
    """Print sample summaries from each model."""
    print("\n")
    print("=" * 100)
    print("  📝 SAMPLE SUMMARIES COMPARISON")
    print("=" * 100)
    
    # Show first sample from each model
    sample_name = BENCHMARK_DATA[0]["name"]
    reference = BENCHMARK_DATA[0]["reference"]
    
    print(f"\n📄 Sample: {sample_name}")
    print(f"\n🎯 REFERENCE SUMMARY:")
    print(f"   {reference}")
    
    for r in all_results:
        if r["samples"]:
            sample = r["samples"][0]
            print(f"\n🤖 {r['model_name']}:")
            print(f"   {sample['generated_summary']}")
            print(f"   [ROUGE-2: {sample['rouge2_f1']:.3f} | BLEU-4: {sample['bleu4']:.3f}]")


def main():
    print("\n" + "=" * 100)
    print("  🚀 SUMMARIZATION MODEL BENCHMARK")
    print("  Testing multiple models on standardized evaluation data")
    print("=" * 100)
    
    device = get_device()
    print(f"\n💻 Device: {device}")
    print(f"📊 Test samples: {len(BENCHMARK_DATA)}")
    print(f"🤖 Models to test: {len(MODELS)}")
    
    all_results = []
    
    for model_info in MODELS:
        result = benchmark_model(model_info, BENCHMARK_DATA, device)
        if result:
            all_results.append(result)
    
    if all_results:
        print_comparison_table(all_results)
        print_sample_summaries(all_results)
    else:
        print("\n❌ No models were successfully tested.")
    
    print("\n✅ Benchmark complete!\n")


if __name__ == "__main__":
    main()
