/*
===========================================
   CONTEXT-AWARE TEXT SUMMARIZER
   Main JavaScript - Handles all interactions
===========================================
*/

// ========== STATE VARIABLES ==========
let selectedSize = 'medium';
let selectedContext = 'balanced';
let uploadedDocumentText = null;  // Store uploaded document text (not shown in textarea)


// ========== INITIALIZATION ==========
document.addEventListener('DOMContentLoaded', function() {
    console.log('Text Summarizer initialized');
    initSizeButtons();
    initContextButtons();
    initFileUpload();
});


// ========== BUTTON HANDLERS ==========

/**
 * Initialize summary size option buttons
 * Updates selectedSize when user clicks a size option
 */
function initSizeButtons() {
    document.querySelectorAll('.option-buttons').forEach(group => {
        group.querySelectorAll('.option-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active from all buttons in group
                group.querySelectorAll('.option-btn').forEach(b => b.classList.remove('active'));
                // Add active to clicked button
                btn.classList.add('active');
                // Update state - use replaceAll for multiple spaces
                selectedSize = btn.textContent.toLowerCase().trim().replace(/\s+/g, '_');
                console.log('Size selected:', selectedSize);
            });
        });
    });
}

/**
 * Initialize context level buttons
 * Updates selectedContext when user clicks a context option
 */
function initContextButtons() {
    document.querySelectorAll('.context-options').forEach(group => {
        group.querySelectorAll('.context-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active from all buttons in group
                group.querySelectorAll('.context-btn').forEach(b => b.classList.remove('active'));
                // Add active to clicked button
                btn.classList.add('active');
                // Update state
                selectedContext = btn.textContent.toLowerCase().replace(' context', '');
                console.log('Context selected:', selectedContext);
            });
        });
    });
}

/**
 * Initialize file upload handler
 * Stores document content but doesn't display in textarea
 */
function initFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const inputText = document.getElementById('inputText');
    
    // Clear uploaded document when user types in textarea
    if (inputText) {
        inputText.addEventListener('input', function() {
            if (uploadedDocumentText !== null) {
                uploadedDocumentText = null;
                hideFileIndicator();
                console.log('Uploaded document cleared - using pasted text');
            }
        });
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                console.log('File selected:', file.name, 'Type:', file.type);
                
                // Show file indicator immediately for all files
                showFileIndicator(file.name);
                
                const extension = file.name.split('.').pop().toLowerCase();
                
                // Handle different file types
                if (['txt', 'md', 'rtf'].includes(extension)) {
                    // Read text files directly - store but don't show
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        uploadedDocumentText = e.target.result;
                        document.getElementById('inputText').value = '';  // Clear textarea
                        document.getElementById('inputText').placeholder = '📄 Document uploaded: ' + file.name + '\n\nClick "Summarize" to process, or paste new text to replace.';
                        console.log('File loaded successfully (stored, not displayed)');
                    };
                    reader.onerror = function() {
                        alert('Error reading file. Please try again.');
                        hideFileIndicator();
                    };
                    reader.readAsText(file);
                } else if (['pdf', 'doc', 'docx'].includes(extension)) {
                    // Upload to server for processing
                    uploadFileToServer(file);
                } else {
                    // Try to read as text anyway
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        uploadedDocumentText = e.target.result;
                        document.getElementById('inputText').value = '';
                        document.getElementById('inputText').placeholder = '📄 Document uploaded: ' + file.name + '\n\nClick "Summarize" to process, or paste new text to replace.';
                        console.log('File loaded successfully (stored, not displayed)');
                    };
                    reader.onerror = function() {
                        alert('Error reading file. Please try a .txt file.');
                        hideFileIndicator();
                    };
                    reader.readAsText(file);
                }
            }
        });
    }
}

/**
 * Upload file to server for processing (PDF, DOC, etc.)
 * Stores text but doesn't display in textarea
 */
async function uploadFileToServer(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state in placeholder
    const inputText = document.getElementById('inputText');
    inputText.value = '';
    inputText.placeholder = '⏳ Processing document...';
    inputText.disabled = true;
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            uploadedDocumentText = result.text;  // Store but don't display
            inputText.value = '';  // Keep textarea empty
            inputText.placeholder = '📄 Document uploaded: ' + file.name + '\n\nClick "Summarize" to process, or paste new text to replace.';
            console.log('File processed successfully (stored, not displayed)');
        } else {
            inputText.value = '';
            inputText.placeholder = 'Paste your text here or upload a document...';
            alert(result.error || 'Error processing file');
            hideFileIndicator();
        }
    } catch (error) {
        console.error('Upload error:', error);
        inputText.value = '';
        inputText.placeholder = 'Paste your text here or upload a document...';
        alert('Error uploading file. Please try again.');
        hideFileIndicator();
    } finally {
        inputText.disabled = false;
    }
}

/**
 * Hide file indicator
 */
function hideFileIndicator() {
    const indicator = document.getElementById('fileIndicator');
    const fileInput = document.getElementById('fileInput');
    if (indicator) indicator.style.display = 'none';
    if (fileInput) fileInput.value = '';
}

/**
 * Show file indicator with filename after upload
 */
function showFileIndicator(filename) {
    const indicator = document.getElementById('fileIndicator');
    const fileNameSpan = document.getElementById('fileName');
    if (indicator && fileNameSpan) {
        fileNameSpan.textContent = filename;
        indicator.style.display = 'flex';
    }
}

/**
 * Remove uploaded file and hide indicator
 */
function removeFile() {
    const indicator = document.getElementById('fileIndicator');
    const fileInput = document.getElementById('fileInput');
    const inputText = document.getElementById('inputText');
    
    if (indicator) indicator.style.display = 'none';
    if (fileInput) fileInput.value = '';
    if (inputText) {
        inputText.value = '';
        inputText.placeholder = 'Paste your text here or upload a document...';
    }
    
    // Clear stored document text
    uploadedDocumentText = null;
    
    console.log('File removed');
}


// ========== TEXT ACTIONS ==========

/**
 * Paste text from clipboard
 */
async function pasteText() {
    try {
        const text = await navigator.clipboard.readText();
        document.getElementById('inputText').value = text;
        console.log('Text pasted from clipboard');
    } catch (err) {
        console.error('Paste failed:', err);
        alert('Failed to paste. Please use Ctrl+V (Cmd+V on Mac) instead.');
    }
}

/**
 * Clear the input textarea
 */
function clearText() {
    document.getElementById('inputText').value = '';
    document.getElementById('inputText').placeholder = 'Paste your text here or upload a document...';
    
    // Clear uploaded document text
    uploadedDocumentText = null;
    
    // Also hide file indicator if visible
    const indicator = document.getElementById('fileIndicator');
    const fileInput = document.getElementById('fileInput');
    if (indicator) indicator.style.display = 'none';
    if (fileInput) fileInput.value = '';
    console.log('Text cleared');
}

/**
 * Copy summary to clipboard
 */
function copySummary() {
    const summaryText = document.getElementById('summaryText').textContent;
    navigator.clipboard.writeText(summaryText).then(() => {
        console.log('Summary copied to clipboard');
        alert('Summary copied to clipboard!');
    }).catch(err => {
        console.error('Copy failed:', err);
    });
}

/**
 * Download summary as a .txt file
 */
function downloadSummary() {
    const summaryText = document.getElementById('summaryText').textContent;
    const blob = new Blob([summaryText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'summary.txt';
    a.click();
    URL.revokeObjectURL(url);
    console.log('Summary downloaded');
}


// ========== MAIN SUMMARIZE FUNCTION ==========

/**
 * Generate summary by calling the backend API
 * This is the main function triggered by the "Generate Summary" button
 * Uses uploaded document text if available, otherwise uses textarea text
 */
async function generateSummary() {
    // Get DOM elements
    const textareaText = document.getElementById('inputText').value.trim();
    const btn = document.getElementById('generateBtn');
    const summaryText = document.getElementById('summaryText');
    const statsBar = document.getElementById('statsBar');
    const statsText = document.getElementById('statsText');
    
    // Use uploaded document text if available, otherwise use textarea
    const text = uploadedDocumentText || textareaText;

    // Validate input
    if (!text || text.length < 50) {
        alert('Please enter at least 50 characters of text or upload a document.');
        return;
    }

    console.log('Generating summary...');
    console.log('   Size:', selectedSize);
    console.log('   Context:', selectedContext);
    console.log('   Source:', uploadedDocumentText ? 'Uploaded Document' : 'Pasted Text');

    // Show loading state
    btn.innerHTML = '<span class="loading"></span> Generating...';
    btn.disabled = true;
    summaryText.textContent = 'Processing your text with AI summarization...';
    statsBar.style.display = 'none';

    try {
        // Call the API
        const response = await fetch('/api/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                summary_size: selectedSize,
                context_level: selectedContext
            })
        });

        const data = await response.json();

        if (data.success) {
            // Success! Show the summary
            summaryText.textContent = data.final_summary;
            statsBar.style.display = 'block';
            
            // Calculate and show statistics
            const reduction = Math.round((1 - data.summary_length / data.original_length) * 100);
            statsText.textContent = `Original: ${data.original_length} words -> Summary: ${data.summary_length} words (${reduction}% reduction)`;
            
            console.log('Summary generated successfully');
        } else {
            // Error from server
            summaryText.textContent = 'Error: ' + data.error;
            statsBar.style.display = 'none';
            console.error('Server error:', data.error);
        }
    } catch (error) {
        // Network or other error
        summaryText.textContent = 'Error connecting to server. Please make sure the server is running.';
        statsBar.style.display = 'none';
        console.error('Connection error:', error);
    }

    // Reset button state
    btn.innerHTML = 'Generate Summary';
    btn.disabled = false;
}
