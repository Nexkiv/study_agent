/**
 * StudyAgent — Main application JavaScript
 * Handles: tabs, classes, uploads, chat (with markdown), flashcards (with search/pagination), modals, toasts.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentClassId = null;
let currentClassName = null;
let allFlashcards = [];      // full flashcard list for search/pagination
let flashcardPage = 1;
let currentSetId = null;     // selected flashcard set (null = all)
let currentSetName = 'All sets';
let generateController = null;  // AbortController for flashcard generation
const FLASHCARDS_PER_PAGE = 25;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    loadOcrMethods();
    // Backdrop click to dismiss any dialog
    document.querySelectorAll('dialog').forEach(dialog => {
        dialog.addEventListener('click', e => {
            const rect = dialog.getBoundingClientRect();
            const clickedOutside = e.clientX < rect.left || e.clientX > rect.right ||
                                   e.clientY < rect.top || e.clientY > rect.bottom;
            if (clickedOutside) dialog.close();
        });
    });
});

// ---------------------------------------------------------------------------
// Toast Notifications
// ---------------------------------------------------------------------------
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ---------------------------------------------------------------------------
// Tab Switching
// ---------------------------------------------------------------------------
function switchTab(tabName) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.remove('border-blue-600', 'text-blue-600', 'dark:text-blue-400', 'dark:border-blue-400', 'bg-white', 'dark:bg-gray-800');
        b.classList.add('border-transparent', 'text-gray-500', 'dark:text-gray-400');
    });
    const panel = document.getElementById(`panel-${tabName}`);
    if (panel) panel.classList.remove('hidden');
    const btn = document.getElementById(`tab-${tabName}`);
    if (btn) {
        btn.classList.remove('border-transparent', 'text-gray-500', 'dark:text-gray-400');
        btn.classList.add('border-blue-600', 'text-blue-600', 'dark:text-blue-400', 'dark:border-blue-400', 'bg-white', 'dark:bg-gray-800');
    }
}

// ---------------------------------------------------------------------------
// Custom Class Dropdown
// ---------------------------------------------------------------------------
function toggleClassDropdown() {
    const list = document.getElementById('class-dropdown-list');
    const btn = document.getElementById('class-dropdown-btn');
    const isOpen = !list.classList.contains('hidden');
    if (isOpen) {
        list.classList.add('hidden');
        btn.setAttribute('aria-expanded', 'false');
    } else {
        list.classList.remove('hidden');
        btn.setAttribute('aria-expanded', 'true');
    }
}

function selectClass(li) {
    const value = li.dataset.value;
    const name = li.dataset.name;
    document.getElementById('class-dropdown-label').textContent = name;
    document.getElementById('class-dropdown-label').classList.remove('text-gray-400', 'dark:text-gray-500');
    document.getElementById('class-dropdown-label').classList.add('text-gray-900', 'dark:text-gray-100');
    document.getElementById('class-dropdown-list').classList.add('hidden');
    document.getElementById('class-dropdown-btn').setAttribute('aria-expanded', 'false');
    // Mark selected
    document.querySelectorAll('#class-dropdown-list li').forEach(item => {
        item.classList.remove('bg-blue-50', 'dark:bg-gray-700');
        item.removeAttribute('aria-selected');
    });
    li.classList.add('bg-blue-50', 'dark:bg-gray-700');
    li.setAttribute('aria-selected', 'true');
    onClassChange(value);
}

// Close dropdown on outside click
document.addEventListener('click', e => {
    const dropdown = document.getElementById('class-dropdown');
    if (dropdown && !dropdown.contains(e.target)) {
        document.getElementById('class-dropdown-list').classList.add('hidden');
        document.getElementById('class-dropdown-btn').setAttribute('aria-expanded', 'false');
    }
});

// Keyboard navigation for dropdown
document.addEventListener('keydown', e => {
    const list = document.getElementById('class-dropdown-list');
    if (!list || list.classList.contains('hidden')) return;
    const items = Array.from(list.querySelectorAll('li'));
    if (!items.length) return;
    const focused = list.querySelector('li.dropdown-focused');
    let idx = focused ? items.indexOf(focused) : -1;
    if (e.key === 'ArrowDown') { e.preventDefault(); idx = Math.min(idx + 1, items.length - 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); idx = Math.max(idx - 1, 0); }
    else if (e.key === 'Enter' && focused) { e.preventDefault(); selectClass(focused); return; }
    else if (e.key === 'Escape') { toggleClassDropdown(); return; }
    else return;
    items.forEach(i => i.classList.remove('dropdown-focused', 'bg-gray-100', 'dark:bg-gray-700'));
    if (items[idx]) {
        items[idx].classList.add('dropdown-focused', 'bg-gray-100', 'dark:bg-gray-700');
        items[idx].scrollIntoView({ block: 'nearest' });
    }
});

// ---------------------------------------------------------------------------
// Class Management
// ---------------------------------------------------------------------------
function onClassChange(value) {
    if (!value) {
        currentClassId = null;
        currentClassName = null;
        document.getElementById('welcome-banner').classList.remove('hidden');
        document.getElementById('main-content').classList.add('hidden');
        document.getElementById('delete-class-btn').classList.add('hidden');
        document.getElementById('header-subtitle').textContent = 'AI-powered study assistant';
        document.getElementById('class-dropdown-label').textContent = 'Select a class...';
        document.getElementById('class-dropdown-label').classList.add('text-gray-400', 'dark:text-gray-500');
        document.getElementById('class-dropdown-label').classList.remove('text-gray-900', 'dark:text-gray-100');
        return;
    }
    currentClassId = parseInt(value);
    const selectedItem = document.querySelector(`#class-dropdown-list li[data-value="${value}"]`);
    currentClassName = selectedItem ? selectedItem.dataset.name : '';

    document.getElementById('welcome-banner').classList.add('hidden');
    document.getElementById('main-content').classList.remove('hidden');
    document.getElementById('delete-class-btn').classList.remove('hidden');
    document.getElementById('header-subtitle').textContent = currentClassName;

    // Clear form fields on class switch
    document.getElementById('doc-name').value = '';
    document.getElementById('file-input').value = '';
    document.getElementById('selected-file-name').classList.add('hidden');
    clearStatus('upload-status');
    clearStatus('flashcard-status');
    clearStatus('ocr-status');
    clearStatus('export-status');

    loadFileList();
    loadChatHistory();
    loadFlashcardSets();
    loadFileSelectors();
}

function showCreateClassModal() {
    const modal = document.getElementById('create-class-modal');
    document.getElementById('new-class-name').value = '';
    document.getElementById('create-class-error').classList.add('hidden');
    modal.showModal();
    document.getElementById('new-class-name').focus();
}

function showDeleteClassModal() {
    if (!currentClassId) return;
    document.getElementById('delete-class-name-display').textContent = currentClassName;
    document.getElementById('delete-class-modal').showModal();
}

async function createClass() {
    const nameInput = document.getElementById('new-class-name');
    const name = nameInput.value.trim();
    const errorEl = document.getElementById('create-class-error');
    if (!name) {
        errorEl.textContent = 'Please enter a class name.';
        errorEl.classList.remove('hidden');
        return;
    }
    try {
        const res = await fetch('/api/classes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        const data = await res.json();
        if (!res.ok) {
            errorEl.textContent = data.error;
            errorEl.classList.remove('hidden');
            return;
        }
        document.getElementById('create-class-modal').close();
        await reloadClassList(data.id);
        showToast(data.existed ? `Switched to ${name}` : `Class "${name}" created`);
    } catch (e) {
        errorEl.textContent = 'Failed to create class.';
        errorEl.classList.remove('hidden');
    }
}

async function deleteClass() {
    if (!currentClassId) return;
    const name = currentClassName;
    try {
        const res = await fetch(`/api/classes/${currentClassId}`, { method: 'DELETE' });
        if (res.ok) {
            document.getElementById('delete-class-modal').close();
            currentClassId = null;
            currentClassName = null;
            await reloadClassList(null);
            document.getElementById('welcome-banner').classList.remove('hidden');
            document.getElementById('main-content').classList.add('hidden');
            document.getElementById('delete-class-btn').classList.add('hidden');
            document.getElementById('header-subtitle').textContent = 'AI-powered study assistant';
            showToast(`Class "${name}" deleted`);
        }
    } catch (e) {
        console.error('Delete class failed:', e);
    }
}

async function reloadClassList(selectId) {
    try {
        const res = await fetch('/api/classes');
        const classes = await res.json();
        const list = document.getElementById('class-dropdown-list');
        list.innerHTML = '';
        classes.forEach(cls => {
            const li = document.createElement('li');
            li.setAttribute('role', 'option');
            li.dataset.value = cls.id;
            li.dataset.name = cls.name;
            li.onclick = () => selectClass(li);
            li.className = 'px-4 py-2.5 cursor-pointer hover:bg-blue-50 dark:hover:bg-gray-700 text-sm flex items-center justify-between';
            const nameSpan = document.createElement('span');
            nameSpan.textContent = cls.name;
            li.appendChild(nameSpan);
            if (cls.file_count) {
                const countSpan = document.createElement('span');
                countSpan.className = 'text-xs text-gray-400';
                countSpan.textContent = `${cls.file_count} file${cls.file_count !== 1 ? 's' : ''}`;
                li.appendChild(countSpan);
            }
            list.appendChild(li);
        });
        if (selectId) {
            const item = list.querySelector(`li[data-value="${selectId}"]`);
            if (item) selectClass(item);
        }
    } catch (e) {
        console.error('Failed to reload classes:', e);
    }
}

// ---------------------------------------------------------------------------
// File Upload
// ---------------------------------------------------------------------------
function onFileSelected(input) {
    const file = input.files[0];
    const display = document.getElementById('selected-file-name');
    const docName = document.getElementById('doc-name');
    if (file) {
        display.textContent = `Selected: ${file.name}`;
        display.classList.remove('hidden');
        if (!docName.value.trim()) {
            let name = file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' ');
            name = name.replace(/\b\w/g, c => c.toUpperCase());
            docName.value = name;
        }
    } else {
        display.classList.add('hidden');
    }
}

async function handleUpload(event) {
    event.preventDefault();
    if (!currentClassId) { showStatus('upload-status', 'Please select a class first.', 'error'); return false; }

    const fileInput = document.getElementById('file-input');
    const docName = document.getElementById('doc-name').value.trim();
    const docType = document.getElementById('doc-type').value;

    if (!fileInput.files[0]) { showStatus('upload-status', 'Please select a file.', 'error'); return false; }
    if (!docName) { showStatus('upload-status', 'Please enter a document name.', 'error'); return false; }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('name', docName);
    formData.append('type', docType);

    const btn = document.getElementById('upload-btn');
    btn.disabled = true;
    btn.textContent = 'Processing...';
    showStatus('upload-status', 'Uploading and processing...', 'info');

    try {
        const res = await fetch(`/api/classes/${currentClassId}/files`, { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) { showStatus('upload-status', data.error, 'error'); return false; }

        const msg = `Uploaded: ${data.name} — ${data.chars.toLocaleString()} chars, ${data.chunks} chunks indexed.`;
        showStatus('upload-status', msg + (data.needs_ocr ? ' Low extraction quality — try OCR below.' : ''), 'success');
        showToast(`"${data.name}" uploaded and processed`);

        fileInput.value = '';
        document.getElementById('doc-name').value = '';
        document.getElementById('selected-file-name').classList.add('hidden');
        loadFileList();
        loadFileSelectors();
        reloadClassList(currentClassId);
    } catch (e) {
        showStatus('upload-status', 'Upload failed. Please try again.', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Upload & Process';
    }
    return false;
}

// ---------------------------------------------------------------------------
// File Management — per-row delete
// ---------------------------------------------------------------------------
function confirmDeleteFileById(fileId, fileName) {
    document.getElementById('delete-file-name-display').textContent = fileName;
    document.getElementById('delete-file-modal').dataset.fileId = fileId;
    document.getElementById('delete-file-modal').showModal();
}

async function deleteFile() {
    const modal = document.getElementById('delete-file-modal');
    const fileId = modal.dataset.fileId;
    modal.close();
    try {
        const res = await fetch(`/api/files/${fileId}`, { method: 'DELETE' });
        const data = await res.json();
        if (res.ok) {
            showToast(data.message);
            loadFileList();
            loadFileSelectors();
            reloadClassList(currentClassId);
        } else {
            showToast(data.error, 'error');
        }
    } catch (e) {
        showToast('Delete failed.', 'error');
    }
}

async function retryOcr() {
    const fileId = document.getElementById('ocr-file-select').value;
    if (!fileId) { showStatus('ocr-status', 'Please select a file to re-extract.', 'error'); return; }
    const methodRadio = document.querySelector('input[name="ocr-method"]:checked');
    const method = methodRadio ? methodRadio.value : 'tesseract';
    showStatus('ocr-status', 'Running OCR...', 'info');
    try {
        const res = await fetch(`/api/files/${fileId}/ocr`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ method }),
        });
        const data = await res.json();
        if (res.ok) {
            showStatus('ocr-status', `${data.message} — ${data.chars.toLocaleString()} chars, ${data.chunks} chunks.`, 'success');
            showToast('OCR re-extraction complete');
            loadFileList();
        } else {
            showStatus('ocr-status', data.error, 'error');
        }
    } catch (e) {
        showStatus('ocr-status', 'OCR failed.', 'error');
    }
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
async function loadChatHistory() {
    if (!currentClassId) return;
    try {
        const res = await fetch(`/api/classes/${currentClassId}/chat`);
        const messages = await res.json();
        const container = document.getElementById('chat-messages');
        const emptyState = document.getElementById('chat-empty');
        if (messages.length === 0) {
            container.innerHTML = '';
            container.appendChild(emptyState);
            emptyState.classList.remove('hidden');
            return;
        }
        emptyState.classList.add('hidden');
        container.innerHTML = messages.map(msg => renderMessage(msg.role, msg.content, msg.created_at)).join('');
        scrollChatToBottom();
    } catch (e) {
        console.error('Failed to load chat history:', e);
    }
}

function renderMessage(role, content, timestamp) {
    const isUser = role === 'user';
    const align = isUser ? 'justify-end' : 'justify-start';
    const bg = isUser
        ? 'bg-blue-600 text-white'
        : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100';

    let rendered;
    if (isUser) {
        rendered = escapeHtml(content);
    } else {
        try {
            rendered = DOMPurify.sanitize(marked.parse(content));
        } catch (e) {
            rendered = escapeHtml(content);
        }
    }

    const contentClass = isUser ? 'text-sm whitespace-pre-wrap' : 'text-sm chat-markdown';
    const timeStr = formatTimestamp(timestamp);

    return `<div class="flex ${align}">
        <div class="max-w-[80%]">
            <div class="rounded-lg px-4 py-3 ${bg}">
                <div class="${contentClass}">${rendered}</div>
            </div>
            <p class="text-xs text-gray-400 dark:text-gray-500 mt-1 ${isUser ? 'text-right' : ''} px-1">${timeStr}</p>
        </div>
    </div>`;
}

function formatTimestamp(ts) {
    if (!ts) return 'Just now';
    const date = new Date(ts);
    if (isNaN(date.getTime())) return 'Just now';
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const time = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    if (isToday) return time;
    return `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${time}`;
}

function escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function scrollChatToBottom() {
    const container = document.getElementById('chat-messages');
    container.scrollTop = container.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message || !currentClassId) return;

    const container = document.getElementById('chat-messages');
    document.getElementById('chat-empty')?.classList.add('hidden');

    // Optimistic: show user message
    container.insertAdjacentHTML('beforeend', renderMessage('user', message));
    input.value = '';
    scrollChatToBottom();

    // Animated typing indicator
    const typingId = 'typing-indicator';
    container.insertAdjacentHTML('beforeend',
        `<div id="${typingId}" class="flex justify-start">
            <div class="max-w-[80%] rounded-lg px-4 py-3 bg-gray-100 dark:bg-gray-700">
                <span class="typing-dots text-gray-400 dark:text-gray-500"><span></span><span></span><span></span></span>
            </div>
        </div>`
    );
    scrollChatToBottom();

    input.disabled = true;
    document.getElementById('send-btn').disabled = true;

    try {
        const res = await fetch(`/api/classes/${currentClassId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });
        const data = await res.json();
        document.getElementById(typingId)?.remove();
        container.insertAdjacentHTML('beforeend', renderMessage('assistant', data.assistant.content));
        scrollChatToBottom();
    } catch (e) {
        document.getElementById(typingId)?.remove();
        container.insertAdjacentHTML('beforeend', renderMessage('assistant', 'Failed to get response. Please try again.'));
        scrollChatToBottom();
    } finally {
        input.disabled = false;
        document.getElementById('send-btn').disabled = false;
        input.focus();
    }
}

function showClearChatModal() {
    if (!currentClassId) return;
    document.getElementById('clear-chat-modal').showModal();
}

async function clearChat() {
    document.getElementById('clear-chat-modal').close();
    try {
        await fetch(`/api/classes/${currentClassId}/chat`, { method: 'DELETE' });
        loadChatHistory();
        showToast('Chat history cleared');
    } catch (e) {
        console.error('Failed to clear chat:', e);
    }
}

// ---------------------------------------------------------------------------
// Flashcard Sets + Flashcards with search + pagination
// ---------------------------------------------------------------------------

async function loadFlashcardSets() {
    if (!currentClassId) return;
    currentSetId = null;
    currentSetName = 'All sets';
    try {
        const res = await fetch(`/api/classes/${currentClassId}/flashcard-sets`);
        const sets = await res.json();
        const list = document.getElementById('set-dropdown-list');
        const container = document.getElementById('flashcard-sets-container');

        // Build custom dropdown options
        list.innerHTML = '';
        // "All sets" option
        const allLi = document.createElement('li');
        allLi.setAttribute('role', 'option');
        allLi.dataset.value = '';
        allLi.dataset.name = 'All sets';
        allLi.onclick = () => selectSet(allLi);
        allLi.className = 'px-3 py-2 cursor-pointer hover:bg-blue-50 dark:hover:bg-gray-700 text-sm bg-blue-50 dark:bg-gray-700';
        allLi.setAttribute('aria-selected', 'true');
        allLi.textContent = 'All sets';
        list.appendChild(allLi);

        sets.forEach(s => {
            const li = document.createElement('li');
            li.setAttribute('role', 'option');
            li.dataset.value = s.id;
            li.dataset.name = s.name;
            li.onclick = () => selectSet(li);
            li.className = 'px-3 py-2 cursor-pointer hover:bg-blue-50 dark:hover:bg-gray-700 text-sm flex items-center justify-between';
            const nameSpan = document.createElement('span');
            nameSpan.textContent = s.name;
            li.appendChild(nameSpan);
            const countSpan = document.createElement('span');
            countSpan.className = 'text-xs text-gray-400';
            countSpan.textContent = `${s.card_count}`;
            li.appendChild(countSpan);
            list.appendChild(li);
        });

        document.getElementById('set-dropdown-label').textContent = 'All sets';
        container.classList.toggle('hidden', sets.length === 0);
        document.getElementById('delete-set-btn').classList.add('hidden');
        updateExportButtonText();

        await loadFlashcards();
    } catch (e) {
        console.error('Failed to load flashcard sets:', e);
    }
}

function toggleSetDropdown() {
    const list = document.getElementById('set-dropdown-list');
    const btn = document.getElementById('set-dropdown-btn');
    const isOpen = !list.classList.contains('hidden');
    list.classList.toggle('hidden');
    btn.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
}

function selectSet(li) {
    const value = li.dataset.value;
    currentSetName = li.dataset.name;
    document.getElementById('set-dropdown-label').textContent = currentSetName;
    document.getElementById('set-dropdown-list').classList.add('hidden');
    document.getElementById('set-dropdown-btn').setAttribute('aria-expanded', 'false');
    // Mark selected
    document.querySelectorAll('#set-dropdown-list li').forEach(item => {
        item.classList.remove('bg-blue-50', 'dark:bg-gray-700');
        item.removeAttribute('aria-selected');
    });
    li.classList.add('bg-blue-50', 'dark:bg-gray-700');
    li.setAttribute('aria-selected', 'true');

    currentSetId = value ? parseInt(value) : null;
    document.getElementById('delete-set-btn').classList.toggle('hidden', !currentSetId);
    updateExportButtonText();
    loadFlashcards();
}

// Close set dropdown on outside click
document.addEventListener('click', e => {
    const dropdown = document.getElementById('set-dropdown');
    if (dropdown && !dropdown.contains(e.target)) {
        document.getElementById('set-dropdown-list').classList.add('hidden');
        document.getElementById('set-dropdown-btn').setAttribute('aria-expanded', 'false');
    }
});

function updateExportButtonText() {
    const btn = document.getElementById('export-btn-text');
    if (!btn) return;
    btn.textContent = currentSetId ? `Export: ${currentSetName}` : 'Export All Saved';
}

function confirmDeleteSet() {
    if (!currentSetId) return;
    document.getElementById('delete-set-name-display').textContent = currentSetName;
    document.getElementById('delete-set-modal').showModal();
}

async function deleteSet() {
    document.getElementById('delete-set-modal').close();
    if (!currentSetId) return;
    try {
        const res = await fetch(`/api/flashcard-sets/${currentSetId}`, { method: 'DELETE' });
        const data = await res.json();
        if (res.ok) {
            showToast(data.message);
            currentSetId = null;
            await loadFlashcardSets();
        } else {
            showToast(data.error, 'error');
        }
    } catch (e) {
        showToast('Failed to delete set.', 'error');
    }
}

async function loadFlashcards() {
    if (!currentClassId) return;
    try {
        let url = `/api/classes/${currentClassId}/flashcards`;
        if (currentSetId) url += `?set_id=${currentSetId}`;
        const res = await fetch(url);
        allFlashcards = await res.json();
        flashcardPage = 1;
        renderCurrentFlashcards();
    } catch (e) {
        console.error('Failed to load flashcards:', e);
    }
}

function renderCurrentFlashcards() {
    const container = document.getElementById('flashcard-table-container');
    const countEl = document.getElementById('flashcard-total-count');
    const searchContainer = document.getElementById('flashcard-search-container');
    const pagination = document.getElementById('flashcard-pagination');

    // Apply search filter
    const searchTerm = (document.getElementById('flashcard-search')?.value || '').toLowerCase();
    let filtered = allFlashcards;
    if (searchTerm) {
        filtered = allFlashcards.filter(fc =>
            fc.term.toLowerCase().includes(searchTerm) ||
            fc.definition.toLowerCase().includes(searchTerm)
        );
    }

    // Disable export when no flashcards
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) exportBtn.disabled = allFlashcards.length === 0;

    if (allFlashcards.length === 0) {
        container.innerHTML = '<p class="text-sm text-gray-500 dark:text-gray-400 italic">No flashcards yet. Generate some above.</p>';
        countEl.textContent = '';
        searchContainer.classList.add('hidden');
        pagination.classList.add('hidden');
        return;
    }

    countEl.textContent = `${allFlashcards.length} saved`;
    searchContainer.classList.remove('hidden');

    // Pagination
    const totalPages = Math.ceil(filtered.length / FLASHCARDS_PER_PAGE);
    if (flashcardPage > totalPages) flashcardPage = Math.max(1, totalPages);
    const start = (flashcardPage - 1) * FLASHCARDS_PER_PAGE;
    const pageItems = filtered.slice(start, start + FLASHCARDS_PER_PAGE);

    container.innerHTML = renderFlashcardTable(pageItems);

    if (filtered.length > FLASHCARDS_PER_PAGE) {
        pagination.classList.remove('hidden');
        document.getElementById('page-info').textContent = `Page ${flashcardPage} of ${totalPages} (${filtered.length} cards)`;
        document.getElementById('prev-page-btn').disabled = flashcardPage <= 1;
        document.getElementById('next-page-btn').disabled = flashcardPage >= totalPages;
    } else {
        pagination.classList.add('hidden');
    }
}

function filterFlashcards(query) {
    flashcardPage = 1;
    renderCurrentFlashcards();
}

function changePage(delta) {
    flashcardPage += delta;
    renderCurrentFlashcards();
    // Scroll to top of flashcard section
    document.getElementById('flashcard-table-container').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderFlashcardTable(flashcards) {
    if (flashcards.length === 0) {
        return '<p class="text-sm text-gray-500 dark:text-gray-400 italic">No matching flashcards.</p>';
    }
    let html = `<table class="w-full text-sm">
        <thead>
            <tr class="border-b border-gray-200 dark:border-gray-700">
                <th class="text-left py-2 px-3 font-medium text-gray-700 dark:text-gray-300 w-[28%]">Term</th>
                <th class="text-left py-2 px-3 font-medium text-gray-700 dark:text-gray-300">Definition</th>
                <th class="w-16"></th>
            </tr>
        </thead>
        <tbody>`;
    flashcards.forEach(fc => {
        const term = escapeHtml(fc.term);
        const def = escapeHtml(fc.definition);
        html += `<tr class="border-b border-gray-100 dark:border-gray-700/50 hover:bg-gray-50 dark:hover:bg-gray-700/50" data-id="${fc.id}" id="fc-row-${fc.id}">
            <td class="py-2 px-3 align-top font-medium" id="fc-term-${fc.id}">${term}</td>
            <td class="py-2 px-3 align-top" id="fc-def-${fc.id}" title="${def}">${def}</td>
            <td class="py-2 px-1 align-top text-right whitespace-nowrap">
                <button onclick="startEditFlashcard(${fc.id})" title="Edit" class="p-1 text-gray-300 dark:text-gray-600 hover:text-blue-500 transition-colors rounded">
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"/></svg>
                </button>
                <button onclick="deleteFlashcard(${fc.id})" title="Delete" class="p-1 text-gray-300 dark:text-gray-600 hover:text-red-500 transition-colors rounded">
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
                </button>
            </td>
        </tr>`;
    });
    html += '</tbody></table>';
    return html;
}

async function deleteFlashcard(id) {
    if (!confirm('Delete this flashcard?')) return;
    try {
        const res = await fetch(`/api/flashcards/${id}`, { method: 'DELETE' });
        if (res.ok) {
            allFlashcards = allFlashcards.filter(fc => fc.id !== id);
            renderCurrentFlashcards();
            showToast('Flashcard deleted');
            // Update set counts while preserving current selection
            const savedSetId = currentSetId;
            await loadFlashcardSets();
            if (savedSetId) {
                const item = document.querySelector(`#set-dropdown-list li[data-value="${savedSetId}"]`);
                if (item) selectSet(item);
            }
        } else {
            showToast('Failed to delete flashcard.', 'error');
        }
    } catch (e) {
        showToast('Failed to delete flashcard.', 'error');
    }
}

function startEditFlashcard(id) {
    const termEl = document.getElementById(`fc-term-${id}`);
    const defEl = document.getElementById(`fc-def-${id}`);
    if (!termEl || !defEl) return;
    const fc = allFlashcards.find(f => f.id === id);
    if (!fc) return;

    termEl.innerHTML = `<input type="text" value="${escapeHtml(fc.term)}" id="fc-edit-term-${id}" class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm">`;
    defEl.innerHTML = `<textarea id="fc-edit-def-${id}" rows="3" class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm">${escapeHtml(fc.definition)}</textarea>`;

    // Replace action buttons with save/cancel
    const row = document.getElementById(`fc-row-${id}`);
    const actionsCell = row.querySelector('td:last-child');
    actionsCell.innerHTML = `
        <button onclick="saveEditFlashcard(${id})" title="Save" class="p-1 text-green-500 hover:text-green-700 transition-colors rounded">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
        </button>
        <button onclick="cancelEditFlashcard(${id})" title="Cancel" class="p-1 text-gray-400 hover:text-gray-600 transition-colors rounded">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
    `;
}

async function saveEditFlashcard(id) {
    const termInput = document.getElementById(`fc-edit-term-${id}`);
    const defInput = document.getElementById(`fc-edit-def-${id}`);
    if (!termInput || !defInput) return;

    const term = termInput.value.trim();
    const definition = defInput.value.trim();
    if (!term || !definition) { showToast('Term and definition cannot be empty.', 'error'); return; }

    try {
        const res = await fetch(`/api/flashcards/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ term, definition }),
        });
        if (res.ok) {
            const updated = await res.json();
            const idx = allFlashcards.findIndex(f => f.id === id);
            if (idx >= 0) allFlashcards[idx] = updated;
            renderCurrentFlashcards();
            showToast('Flashcard updated');
        } else {
            showToast('Failed to save.', 'error');
        }
    } catch (e) {
        showToast('Failed to save.', 'error');
    }
}

function cancelEditFlashcard(id) {
    renderCurrentFlashcards();
}

async function generateFlashcards() {
    if (!currentClassId) return;
    const topic = document.getElementById('flashcard-topic').value.trim();
    const btn = document.getElementById('generate-btn');
    const cancelBtn = document.getElementById('cancel-generate-btn');
    btn.classList.add('hidden');
    cancelBtn.classList.remove('hidden');
    showStatus('flashcard-status', 'Generating flashcards — this may take a moment...', 'info');

    generateController = new AbortController();

    try {
        const res = await fetch(`/api/classes/${currentClassId}/flashcards`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic }),
            signal: generateController.signal,
        });
        const data = await res.json();
        if (!res.ok) { showStatus('flashcard-status', data.error, 'error'); return; }

        if (data.cancelled) {
            showStatus('flashcard-status', 'Generation cancelled.', 'info');
            showToast('Generation cancelled');
            return;
        }

        let msg = `${data.generated} flashcards generated from ${data.class_name} materials.`;
        if (data.total > data.generated) msg += ` (${data.total} total saved)`;
        showStatus('flashcard-status', msg, 'success');
        showToast(`${data.generated} flashcards generated`);

        // Reload sets and select the new one
        await loadFlashcardSets();
        if (data.set_id) {
            const item = document.querySelector(`#set-dropdown-list li[data-value="${data.set_id}"]`);
            if (item) selectSet(item);
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            showStatus('flashcard-status', 'Generation cancelled.', 'info');
            showToast('Generation cancelled');
        } else {
            showStatus('flashcard-status', 'Something went wrong. Please try again.', 'error');
        }
    } finally {
        generateController = null;
        btn.classList.remove('hidden');
        cancelBtn.classList.add('hidden');
    }
}

async function cancelGeneration() {
    fetch('/api/flashcards/cancel', { method: 'POST' });
    if (generateController) generateController.abort();
}

async function exportFlashcards() {
    if (!currentClassId) return;
    const format = document.querySelector('input[name="export-format"]:checked')?.value || 'quizlet';
    try {
        let exportUrl = `/api/classes/${currentClassId}/export?format=${format}`;
        if (currentSetId) exportUrl += `&set_id=${currentSetId}`;
        const res = await fetch(exportUrl);
        if (!res.ok) {
            const data = await res.json();
            showStatus('export-status', data.error, 'error');
            return;
        }
        const blob = await res.blob();
        const disposition = res.headers.get('Content-Disposition') || '';
        const match = disposition.match(/filename="?(.+?)"?$/);
        const filename = match ? match[1] : `flashcards.${format === 'anki' ? 'csv' : 'tsv'}`;
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(blobUrl);
        showStatus('export-status', `Exported as ${filename}`, 'success');
        showToast(`Flashcards exported`);
    } catch (e) {
        showStatus('export-status', 'Export failed.', 'error');
    }
}

// ---------------------------------------------------------------------------
// Status Messages
// ---------------------------------------------------------------------------
function showStatus(elementId, message, type) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const colors = {
        success: 'text-green-600 dark:text-green-400',
        error: 'text-red-600 dark:text-red-400',
        info: 'text-blue-600 dark:text-blue-400',
    };
    el.className = `text-sm ${colors[type] || ''}`;
    el.textContent = message;
}

function clearStatus(elementId) {
    const el = document.getElementById(elementId);
    if (el) { el.textContent = ''; el.className = 'text-sm'; }
}

// ---------------------------------------------------------------------------
// Data Loading
// ---------------------------------------------------------------------------
async function loadFileList() {
    if (!currentClassId) return;
    try {
        const res = await fetch(`/partials/materials/${currentClassId}`);
        document.getElementById('file-list-container').innerHTML = await res.text();
    } catch (e) {
        console.error('Failed to load file list:', e);
    }
}

async function loadFileSelectors() {
    if (!currentClassId) return;
    try {
        const res = await fetch(`/api/classes/${currentClassId}/inputs`);
        const inputs = await res.json();
        const ocrSelect = document.getElementById('ocr-file-select');
        ocrSelect.innerHTML = '<option value="">Choose a file...</option>';
        inputs.forEach(inp => {
            ocrSelect.innerHTML += `<option value="${inp.id}">${inp.name} (${inp.type})</option>`;
        });
    } catch (e) {
        console.error('Failed to load file selectors:', e);
    }
}

async function loadOcrMethods() {
    try {
        const res = await fetch('/api/ocr-methods');
        const methods = await res.json();
        const container = document.getElementById('ocr-methods');
        if (!container) return;
        container.innerHTML = methods.map((m, i) =>
            `<label class="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="ocr-method" value="${m}" ${i === 0 ? 'checked' : ''}
                    class="text-blue-600 focus:ring-blue-500">
                <span class="text-sm">${m.charAt(0).toUpperCase() + m.slice(1)}</span>
            </label>`
        ).join('');
    } catch (e) {
        console.error('Failed to load OCR methods:', e);
    }
}
