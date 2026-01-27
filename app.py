"""
Flask Web Application - Main Entry Point
Handles all HTTP routes and serves the web interface

Uses modular backend:
- backend/chunking: Text splitting
- backend/processing: ML model inference
- backend/io: File handling
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from backend import HybridSummarizer, FileHandler
from werkzeug.utils import secure_filename
import os
import tempfile

app = Flask(__name__, 
            template_folder='templates',
            static_folder='frontend')
CORS(app)

# Initialize components
print("Initializing Hybrid Summarizer...")
summarizer = HybridSummarizer()
file_handler = FileHandler()
print("Summarizer ready!")


# ==================== WEB ROUTES ====================

@app.route('/')
def home():
    """Serve the main web page"""
    return render_template('index.html')


# ==================== API ROUTES ====================

@app.route('/api/summarize', methods=['POST'])
def summarize():
    """
    API endpoint to generate summary
    
    Request JSON:
        - text: The text to summarize
        - summary_size: very_small, small, medium, large, very_large
        - context_level: simple, balanced, detailed
    
    Returns JSON:
        - success: Boolean
        - final_summary: The generated summary
        - extracted_summary: Intermediate extractive summary
        - original_length: Word count of original text
        - summary_length: Word count of summary
    """
    try:
        data = request.json
        text = data.get('text', '')
        summary_size = data.get('summary_size', 'medium')
        context_level = data.get('context_level', 'balanced')
        
        # Validate input
        if not text or len(text.strip()) < 50:
            return jsonify({
                'success': False,
                'error': 'Please provide at least 50 characters of text.'
            }), 400
        
        # Generate summary
        result = summarizer.hybrid_summarize(text, summary_size, context_level)
        
        return jsonify({
            'success': True,
            'extracted_summary': result['extracted'],
            'final_summary': result['final_summary'],
            'original_length': len(text.split()),
            'summary_length': len(result['final_summary'].split())
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health')
def health_check():
    """API endpoint to check if server is running"""
    return jsonify({
        'status': 'ok', 
        'model': summarizer.model_name,
        'message': 'Text Summarizer is running!'
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    API endpoint to upload and extract text from files (PDF, DOC, DOCX)
    Uses the modular FileHandler from backend/io
    
    Returns JSON:
        - success: Boolean
        - text: Extracted text from the file
        - error: Error message if failed
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Use FileHandler to extract text
            success, result = file_handler.extract_text(tmp_path)
            
            if success:
                return jsonify({
                    'success': True,
                    'text': result
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result
                }), 400
        
        finally:
            # Clean up temp file
            file_handler.cleanup_temp_file(tmp_path)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status')
def get_status():
    """API endpoint to get system status and supported formats"""
    return jsonify({
        'status': 'ok',
        'summarizer': summarizer.get_status(),
        'supported_formats': file_handler.get_supported_formats()
    })


# ==================== RUN SERVER ====================

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  CONTEXT-AWARE TEXT SUMMARIZER")
    print("  Open http://localhost:5002 in your browser")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5002, host='0.0.0.0')
