# Context-Aware Text Summarizer

A hybrid text summarization web application combining **Extractive** (TextRank) and **Abstractive** (mT5) techniques.

---

## 📁 Project Structure

```
text_summarizer_app/
│
├── app.py                      # 🚀 Main Flask application (entry point)
├── requirements.txt            # 📦 Python dependencies
├── README.md                   # 📖 This documentation file
│
├── backend/                    # 🧠 Backend Logic (AI/ML)
│   ├── __init__.py
│   ├── config.py               # ⚙️ Configuration settings
│   └── hybrid_summarizer.py    # 🤖 Hybrid summarization logic
│
├── frontend/                   # 🎨 Frontend Assets
│   ├── css/
│   │   └── style.css           # 💅 All styles
│   └── js/
│       └── main.js             # ⚡ Frontend JavaScript
│
└── templates/                  # 📄 HTML Templates
    └── index.html              # 🏠 Main page template
```

---

## 🚀 Quick Start

### 1. Navigate to the project folder
```bash
cd text_summarizer_app
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 5. Run the application
```bash
python app.py
```

### 6. Open in browser
Navigate to **http://localhost:5001**

---

## 🔧 Features

| Feature | Description |
|---------|-------------|
| **Hybrid Summarization** | Combines TextRank + mT5 for best results |
| **Adjustable Size** | Very Small → Very Large options |
| **Context Levels** | Simple, Balanced, Detailed |
| **File Upload** | Upload .txt files |
| **Copy & Download** | Easy export options |
| **Statistics** | Word count & reduction % |

---

## 🤖 How It Works

### Step 1: Extractive (TextRank)
- Identifies the most important sentences
- Uses graph-based ranking algorithm
- Preserves original wording

### Step 2: Abstractive (mT5)
- Rewrites extracted text
- Generates new, coherent sentences
- Produces fluent summary

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/summarize` | POST | Generate summary |
| `/api/health` | GET | Health check |

### POST /api/summarize

**Request:**
```json
{
    "text": "Your text to summarize...",
    "summary_size": "medium",
    "context_level": "balanced"
}
```

**Response:**
```json
{
    "success": true,
    "final_summary": "Generated summary...",
    "extracted_summary": "Intermediate extractive result...",
    "original_length": 500,
    "summary_length": 50
}
```

---

## 📝 License

Educational/Research purposes.
