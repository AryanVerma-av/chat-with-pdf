// API Configuration
const API_BASE_URL = (window.location.port === '8000') 
  ? '' 
  : 'http://127.0.0.1:8000';

// DOM Elements
const errorBanner = document.getElementById('errorBanner');
const errorMessage = document.getElementById('errorMessage');
const dismissErrorBtn = document.getElementById('dismissErrorBtn');

const dropZone = document.getElementById('dropZone');
const pdfFileInput = document.getElementById('pdfFileInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadStatusText = document.getElementById('uploadStatusText');

const uploadedInfo = document.getElementById('uploadedInfo');
const uploadedFilename = document.getElementById('uploadedFilename');
const uploadedMetadata = document.getElementById('uploadedMetadata');
const clearDocBtn = document.getElementById('clearDocBtn');

const questionForm = document.getElementById('questionForm');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const askBtnText = document.getElementById('askBtnText');
const askSpinner = document.getElementById('askSpinner');

const answerSection = document.getElementById('answerSection');
const answerContent = document.getElementById('answerContent');
const sourcesContainer = document.getElementById('sourcesContainer');
const sourcesList = document.getElementById('sourcesList');

// State
let isDocumentLoaded = false;

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  checkBackendStatus();
  setupEventListeners();
});

function setupEventListeners() {
  // File selection
  pdfFileInput.addEventListener('change', handleFileSelect);

  // Drag and drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      uploadPDF(file);
    }
  });

  // Question submission
  questionForm.addEventListener('submit', handleQuestionSubmit);

  // Clear / Change document
  clearDocBtn.addEventListener('click', handleClearDocument);

  // Dismiss error
  dismissErrorBtn.addEventListener('click', hideError);
}

// Error Management
function showError(message) {
  errorMessage.textContent = message;
  errorBanner.classList.remove('hidden');
}

function hideError() {
  errorBanner.classList.add('hidden');
  errorMessage.textContent = '';
}

// Check Backend Health and Active Document
async function checkBackendStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) {
      return;
    }
    const data = await res.json();
    if (data.pdf_loaded && data.active_document) {
      setDocumentLoadedState(data.active_document, data.total_pages, data.total_chunks);
    } else {
      setInitialState();
    }
  } catch (err) {
    // Backend may not be started yet; keep initial state
    setInitialState();
  }
}

// File Selection Handler
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    uploadPDF(file);
  }
}

// Upload PDF to Backend
async function uploadPDF(file) {
  hideError();

  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showError('Selected file is not a PDF. Please choose a .pdf document.');
    return;
  }

  // Set uploading state
  setUploadingState(true, 'Processing PDF and generating embeddings...');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Failed to process PDF.');
    }

    setDocumentLoadedState(data.filename, data.total_pages, data.total_chunks);
  } catch (err) {
    showError(err.message || 'Error occurred while processing PDF.');
    setInitialState();
  } finally {
    setUploadingState(false);
    pdfFileInput.value = '';
  }
}

// Question Submission Handler
async function handleQuestionSubmit(event) {
  event.preventDefault();
  hideError();

  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  if (!isDocumentLoaded) {
    showError('Please upload a PDF document before asking a question.');
    return;
  }

  setAskingState(true);

  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Failed to generate answer.');
    }

    renderAnswer(data.answer, data.sources);
  } catch (err) {
    showError(err.message || 'Error occurred while retrieving answer.');
  } finally {
    setAskingState(false);
  }
}

// Render Answer and Sources
function renderAnswer(answer, sources) {
  answerContent.textContent = answer;
  sourcesList.innerHTML = '';

  if (sources && sources.length > 0) {
    // Deduplicate distinct pages for clean display
    const pageMap = new Map();
    sources.forEach(src => {
      if (!pageMap.has(src.page)) {
        pageMap.set(src.page, src.snippet);
      }
    });

    pageMap.forEach((snippet, page) => {
      const item = document.createElement('div');
      item.className = 'source-item';

      const header = document.createElement('div');
      header.className = 'source-header';
      header.textContent = `Page ${page}`;

      const snippetEl = document.createElement('div');
      snippetEl.className = 'source-snippet';
      snippetEl.textContent = `"${snippet}"`;

      item.appendChild(header);
      item.appendChild(snippetEl);
      sourcesList.appendChild(item);
    });

    sourcesContainer.classList.remove('hidden');
  } else {
    sourcesContainer.classList.add('hidden');
  }

  answerSection.classList.remove('hidden');
}

// Clear Active Document
async function handleClearDocument() {
  hideError();
  try {
    await fetch(`${API_BASE_URL}/clear`, { method: 'POST' });
  } catch (err) {
    // Ignore network error on clear
  }
  setInitialState();
}

// UI State Management Functions
function setInitialState() {
  isDocumentLoaded = false;
  dropZone.classList.remove('hidden');
  uploadedInfo.classList.add('hidden');
  uploadProgress.classList.add('hidden');

  questionInput.value = '';
  questionInput.disabled = true;
  questionInput.placeholder = 'Upload a PDF first to ask questions...';
  askBtn.disabled = true;

  answerSection.classList.add('hidden');
  answerContent.textContent = '';
  sourcesList.innerHTML = '';
}

function setUploadingState(isUploading, message = 'Processing PDF...') {
  if (isUploading) {
    dropZone.classList.add('hidden');
    uploadedInfo.classList.add('hidden');
    uploadProgress.classList.remove('hidden');
    uploadStatusText.textContent = message;
    questionInput.disabled = true;
    askBtn.disabled = true;
  } else {
    uploadProgress.classList.add('hidden');
  }
}

function setDocumentLoadedState(filename, totalPages, totalChunks) {
  isDocumentLoaded = true;
  dropZone.classList.add('hidden');
  uploadedInfo.classList.remove('hidden');

  uploadedFilename.textContent = filename;
  uploadedMetadata.textContent = `${totalPages} page(s) | ${totalChunks} chunk(s) indexed`;

  questionInput.disabled = false;
  questionInput.placeholder = 'Ask a question about the document...';
  askBtn.disabled = false;
  questionInput.focus();
}

function setAskingState(isAsking) {
  if (isAsking) {
    askBtn.disabled = true;
    questionInput.disabled = true;
    askBtnText.classList.add('hidden');
    askSpinner.classList.remove('hidden');
  } else {
    askBtn.disabled = false;
    questionInput.disabled = false;
    askBtnText.classList.remove('hidden');
    askSpinner.classList.add('hidden');
    questionInput.focus();
  }
}
