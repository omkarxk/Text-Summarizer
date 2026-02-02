from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import re
import unicodedata
from transformers import BartTokenizer, BartForConditionalGeneration
import PyPDF2
import io
import base64

app = Flask(__name__)
CORS(app)

# =============================
# CONFIG
# =============================
MODEL_PATH = "./bart_dailymail_final"

# =============================
# LOAD MODEL
# =============================
print("🔄 Loading model...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
model = model.to(DEVICE)
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# =============================
# HELPER FUNCTIONS
# =============================
def _normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _sentence_split(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# =============================
# PURE ABSTRACTIVE SUMMARIZER
# =============================
class AbstractiveSummarizer:
    def __init__(self):
        self.tokenizer = tokenizer
        self.model = model
        self.device = DEVICE

    def _split_into_segments(self, text, max_tokens=900):
        sentences = _sentence_split(text)
        segments = []
        current = []
        current_len = 0
        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent, add_special_tokens=False))
            if current_len + sent_tokens > max_tokens and current:
                segments.append(" ".join(current))
                current = [sent]
                current_len = sent_tokens
            else:
                current.append(sent)
                current_len += sent_tokens
        if current:
            segments.append(" ".join(current))
        return segments

    def _generate(self, text, max_len, min_len):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False
        ).to(self.device)
        gen_params = {
            "num_beams": 4,
            "length_penalty": 1.2,
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 1.2,
            "early_stopping": True,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                **gen_params
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _refine(self, draft, min_len, max_len):
        if not draft or len(draft.split()) < 30:
            return draft
        refine_min = max(40, min_len // 2)
        refine_max = max(refine_min + 20, int(max_len * 0.9))
        return self._generate(draft, max_len=refine_max, min_len=refine_min)

    def summarize(self, text, size="medium"):
        text = _normalize_text(text)
        word_count = len(text.split())
        if word_count < 50:
            return "Error: Input text too short (minimum 50 characters required)"
        # Determine length bounds by size as a proportion of input length
        ratio_map = {
            "short": 0.25,
            "medium": 0.50,
            "long": 0.75
        }
        ratio = ratio_map.get(size, 0.30)
        target_words = max(60, int(word_count * ratio))
        min_words = max(40, int(target_words * 0.9))
        max_words = max(min_words + 20, int(target_words * 1.1))
        # Convert words to tokens (approx.)
        token_ratio = 1.3
        max_len = min(1024, int(max_words * token_ratio))
        min_len = max(40, min(max_len - 20, int(min_words * token_ratio)))
        # Long documents: segment → summarize → merge
        if word_count > 900:
            segments = self._split_into_segments(text, max_tokens=900)
            segment_summaries = []
            for seg in segments:
                segment_summaries.append(self._generate(seg, max_len=220, min_len=120))
            combined = " ".join(segment_summaries)
            draft = self._generate(combined, max_len=max_len, min_len=min_len)
        else:
            draft = self._generate(text, max_len=max_len, min_len=min_len)
        refined = self._refine(draft, min_len=min_len, max_len=max_len)
        return " ".join(refined.split())

summarizer = AbstractiveSummarizer()

def summarize_document(text, summary_size='medium'):
    size_map = {
        'very-small': 'short',
        'small': 'short',
        'medium': 'medium',
        'long': 'long',
        'very-long': 'long'
    }
    mapped_size = size_map.get(summary_size, 'medium')
    return summarizer.summarize(text, size=mapped_size)

# =============================
# EXTRACT TEXT FROM PDF
# =============================
def extract_text_from_pdf(pdf_data):
    """Extract text from PDF bytes"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

# =============================
# API ENDPOINTS
# =============================
@app.route('/')
def index():
    return send_from_directory('.', 'pdf-summarizer.html')

@app.route('/summarize', methods=['POST'])

def summarize():
    try:
        data = request.json
        summary_size = data.get('summary_size', 'medium')
        if 'pdf_data' in data:
            pdf_base64 = data['pdf_data']
            pdf_bytes = base64.b64decode(pdf_base64)
            text = extract_text_from_pdf(pdf_bytes)
        elif 'text' in data:
            text = data['text']
        else:
            return jsonify({'error': 'No text or PDF provided'}), 400
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Empty text provided'}), 400
        summary = summarize_document(text, summary_size)
        return jsonify({
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split())
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)