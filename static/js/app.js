const API_URL = '/summarize';

let currentMode = 'pdf';
let uploadedFile = null;
let summarySize = 'medium';
let contextLevel = 'medium';
let pdfObjectUrl = null;
let currentZoom = 100;

function setSummarySize(size) {
    summarySize = size;
    document.querySelectorAll('[data-size]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.size === size);
    });
}

function setContextLevel(level) {
    contextLevel = level;
    document.querySelectorAll('[data-level]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.level === level);
    });
}

function zoomIn() {
    currentZoom = Math.min(currentZoom + 25, 200);
    updateZoom();
}

function zoomOut() {
    currentZoom = Math.max(currentZoom - 25, 50);
    updateZoom();
}

function updateZoom() {
    document.getElementById('zoomLevel').textContent = currentZoom;
    const iframe = document.getElementById('pdfViewer');
    if (iframe.src) {
        iframe.style.transform = `scale(${currentZoom / 100})`;
        iframe.style.transformOrigin = 'top left';
        iframe.style.width = `${100 / (currentZoom / 100)}%`;
        iframe.style.height = `${100 / (currentZoom / 100)}%`;
    }
}

function switchMode(mode) {
    currentMode = mode;
    
    document.getElementById('pdfModeBtn').classList.toggle('active', mode === 'pdf');
    document.getElementById('textModeBtn').classList.toggle('active', mode === 'text');
    
    document.getElementById('pdfMode').classList.toggle('hidden', mode !== 'pdf');
    document.getElementById('textMode').classList.toggle('hidden', mode !== 'text');
    
    clearSummary();
    updateSummarizeButton();
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
        uploadedFile = file;
        
        if (pdfObjectUrl) {
            URL.revokeObjectURL(pdfObjectUrl);
        }
        
        pdfObjectUrl = URL.createObjectURL(file);
        
        document.getElementById('uploadArea').classList.add('hidden');
        document.getElementById('filePreview').classList.remove('hidden');
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('pdfFilename').textContent = file.name;
        
        const iframe = document.getElementById('pdfViewer');
        const placeholder = document.getElementById('pdfPlaceholder');
        
        iframe.src = pdfObjectUrl;
        iframe.style.display = 'block';
        placeholder.style.display = 'none';
        
        currentZoom = 100;
        updateZoom();
        
        clearSummary();
        updateSummarizeButton();
    }
}

function removeFile() {
    uploadedFile = null;
    
    if (pdfObjectUrl) {
        URL.revokeObjectURL(pdfObjectUrl);
        pdfObjectUrl = null;
    }
    
    document.getElementById('uploadArea').classList.remove('hidden');
    document.getElementById('filePreview').classList.add('hidden');
    document.getElementById('fileInput').value = '';
    
    const iframe = document.getElementById('pdfViewer');
    iframe.src = '';
    iframe.style.display = 'none';
    document.getElementById('pdfPlaceholder').style.display = 'flex';
    
    clearSummary();
    updateSummarizeButton();
}

function updateSummarizeButton() {
    const btn = document.getElementById('summarizeBtn');
    const canSummarize = (currentMode === 'pdf' && uploadedFile) || 
                        (currentMode === 'text' && document.getElementById('textInput').value.trim());
    btn.disabled = !canSummarize;
}

document.getElementById('textInput').addEventListener('input', updateSummarizeButton);

async function summarize() {
    const btn = document.getElementById('summarizeBtn');
    btn.disabled = true;
    btn.innerHTML = `
        <div class="spinner"></div>
        Summarizing...
    `;

    clearSummary();

    try {
        let requestData = {
            summary_size: summarySize,
            context_level: contextLevel
        };

        if (currentMode === 'pdf' && uploadedFile) {
            const base64Data = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result.split(',')[1]);
                reader.onerror = reject;
                reader.readAsDataURL(uploadedFile);
            });
            requestData.pdf_data = base64Data;
        } else if (currentMode === 'text') {
            requestData.text = document.getElementById('textInput').value.trim();
        }

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData),
        });

        const data = await response.json();
        
        if (response.ok) {
            displaySummary(data.summary, data.original_length, data.summary_length);
        } else {
            displaySummary('Error: ' + (data.error || 'Failed to generate summary'));
        }
    } catch (error) {
        console.error('Error summarizing:', error);
        displaySummary('An error occurred while generating the summary. Please make sure the Flask server is running.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            Summarize
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"/>
            </svg>
        `;
        updateSummarizeButton();
    }
}

function displaySummary(text, originalLength = 0, summaryLength = 0) {
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('summaryText').classList.remove('hidden');
    document.getElementById('summaryText').textContent = text;
    document.getElementById('copyBtn').classList.remove('hidden');
    
    if (originalLength > 0 && summaryLength > 0) {
        document.getElementById('statsSection').classList.remove('hidden');
        document.getElementById('originalWords').textContent = originalLength;
        document.getElementById('summaryWords').textContent = summaryLength;
        const compression = Math.round((1 - summaryLength / originalLength) * 100);
        document.getElementById('compression').textContent = compression + '%';
    }
}

function clearSummary() {
    document.getElementById('emptyState').classList.remove('hidden');
    document.getElementById('summaryText').classList.add('hidden');
    document.getElementById('summaryText').textContent = '';
    document.getElementById('copyBtn').classList.add('hidden');
    document.getElementById('statsSection').classList.add('hidden');
}

function copySummary() {
    const text = document.getElementById('summaryText').textContent;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copyBtn');
        const originalHTML = btn.innerHTML;
        btn.innerHTML = `
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
            </svg>
        `;
        setTimeout(() => {
            btn.innerHTML = originalHTML;
        }, 2000);
    });
}
