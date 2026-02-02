# 📝 PDF Summarizer

A modern web application that summarizes PDF documents and text using a fine-tuned BART model. Built with Flask and a beautiful, responsive UI.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.35-orange.svg)

## ✨ Features

- **PDF Upload & Preview**: Upload PDF files and preview them directly in the browser
- **Text Input**: Paste or type text for quick summarization
- **Adjustable Summary Length**: Choose from Very Small, Small, Medium, Long, or Very Long summaries
- **Context Levels**: Easy, Medium, and Hard context options
- **Real-time Stats**: View original word count, summary word count, and compression ratio
- **Modern UI**: Beautiful dark theme with animated gradient background
- **Copy to Clipboard**: One-click copy for generated summaries

## 📁 Project Structure

```
project_root/
├── app.py                    # Flask backend with BART model
├── Requirements.txt          # Python dependencies
├── templates/
│   └── index.html            # Main HTML template
├── static/
│   ├── css/
│   │   └── styles.css        # Application styles
│   └── js/
│       └── app.js            # Frontend JavaScript
└── bart_dailymail_final/     # Fine-tuned BART model files
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── ...
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/omkarxk/Text-Summarizer.git
   cd Text-Summarizer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r Requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:5002`

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the main web interface |
| `/summarize` | POST | Summarizes text or PDF content |
| `/health` | GET | Health check endpoint |

### Summarize Endpoint

**Request Body:**
```json
{
  "text": "Your text to summarize...",
  "summary_size": "medium"
}
```

Or for PDF:
```json
{
  "pdf_data": "<base64_encoded_pdf>",
  "summary_size": "medium"
}
```

**Response:**
```json
{
  "summary": "Summarized text...",
  "original_length": 500,
  "summary_length": 100
}
```

## 🎨 Screenshots

The application features a modern dark theme with:
- Split-panel layout for input and output
- Animated gradient background
- Responsive design for all screen sizes

## 🛠️ Tech Stack

- **Backend**: Flask, Python
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Model**: BART (fine-tuned on CNN/DailyMail)
- **Libraries**: Transformers, PyTorch, PyPDF2

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

Made with ❤️ using BART and Flask
