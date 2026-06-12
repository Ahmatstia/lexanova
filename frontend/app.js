// Konfigurasi API Backend Lokal
const API_URL = 'http://localhost:8000';

// Elemen DOM
const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const typingIndicator = document.getElementById('typing-indicator');
const ragToggle = document.getElementById('rag-toggle');

// Kontrol Monitor
const camStartBtn = document.getElementById('cam-start');
const camStopBtn = document.getElementById('cam-stop');
const camStatus = document.getElementById('cam-status');

// Konfigurasi Marked.js untuk render Markdown dengan highlight syntax
marked.setOptions({
    highlight: function(code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
    },
    breaks: true
});

// Otomatis resize textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Menangani form submit
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Reset input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Tampilkan pesan user
    addMessage(message, 'user');
    
    // Tampilkan loading
    typingIndicator.classList.remove('hidden');
    
    try {
        const response = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                pesan: message,
                use_rag: ragToggle.checked
            })
        });
        
        if (!response.ok) throw new Error('Network response was not ok');
        
        const data = await response.json();
        
        // Sembunyikan loading
        typingIndicator.classList.add('hidden');
        
        // Tampilkan pesan AI
        addMessage(data.lexa_response, 'ai', data);
        
    } catch (error) {
        console.error('Error:', error);
        typingIndicator.classList.add('hidden');
        addMessage('Maaf, saya tidak dapat terhubung ke server backend lokal. Pastikan `uvicorn app.main:app` sudah berjalan.', 'ai', { source: 'error' });
    }
});

// Handle Enter untuk submit (Shift+Enter untuk baris baru)
chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Fungsi untuk menambahkan pesan ke UI
function addMessage(text, sender, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const isUser = sender === 'user';
    const icon = isUser ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    // Waktu saat ini
    const timeString = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    // Ekstra info untuk AI
    let metaString = `Lexa • ${timeString}`;
    if (!isUser && metadata) {
        if (metadata.source === 'os_commander') {
            metaString += ` • ⚡ OS Action (${metadata.processing_time_ms}ms)`;
        } else if (metadata.source === 'rag') {
            metaString += ` • 📚 Study Buddy (${metadata.processing_time_ms}ms)`;
        } else if (metadata.source === 'ollama_direct') {
            metaString += ` • 🧠 Ollama Direct (${metadata.processing_time_ms}ms)`;
        } else {
            metaString += ` • ⚠️ System Error`;
        }
    }
    
    if (isUser) {
        metaString = `You • ${timeString}`;
    }

    // Render text (Gunakan markdown khusus untuk pesan AI)
    const renderText = isUser ? escapeHTML(text) : marked.parse(text);

    messageDiv.innerHTML = `
        <div class="avatar">${icon}</div>
        <div class="bubble">
            <div class="content">${renderText}</div>
            <div class="meta">${metaString}</div>
        </div>
    `;
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Escape HTML ringan untuk mencegah XSS di input user
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// ==========================================
// Kontrol Modul Workspace Monitor (Kamera)
// ==========================================

camStartBtn.addEventListener('click', async () => {
    try {
        camStartBtn.disabled = true;
        camStartBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
        
        const response = await fetch(`${API_URL}/api/monitor/start`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            camStatus.classList.add('active');
            camStatus.querySelector('.text').textContent = 'Kamera Aktif';
            camStopBtn.disabled = false;
        } else {
            alert('Gagal menyalakan kamera: ' + data.message);
            camStartBtn.disabled = false;
        }
    } catch (error) {
        alert('Gagal menghubungi backend untuk menyalakan kamera.');
    } finally {
        camStartBtn.innerHTML = '<i class="fa-solid fa-video"></i> Start';
    }
});

camStopBtn.addEventListener('click', async () => {
    try {
        camStopBtn.disabled = true;
        
        const response = await fetch(`${API_URL}/api/monitor/stop`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            camStatus.classList.remove('active');
            camStatus.querySelector('.text').textContent = 'Kamera Mati';
            camStartBtn.disabled = false;
        }
    } catch (error) {
        alert('Gagal mematikan kamera.');
        camStopBtn.disabled = false;
    }
});

// ==========================================
// Kontrol Upload Dokumen Study Buddy (RAG)
// ==========================================

const uploadBtn = document.getElementById('upload-btn');
const fileUpload = document.getElementById('file-upload');
const uploadStatus = document.getElementById('upload-status');

uploadBtn.addEventListener('click', () => {
    fileUpload.click();
});

fileUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Tampilkan status loading
    uploadStatus.style.display = 'flex';
    uploadStatus.classList.add('active');
    uploadStatus.querySelector('.text').textContent = `Mengunggah...`;
    uploadBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/api/study/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            uploadStatus.querySelector('.text').textContent = 'Upload Sukses!';
            uploadStatus.querySelector('.dot').style.backgroundColor = 'var(--success)';
            
            // Beri tahu user lewat chat UI
            addMessage(`📄 Dokumen **${file.name}** berhasil diunggah. Lexa sedang membacanya di background. Anda sudah bisa mulai bertanya!`, 'ai', { source: 'rag' });
        } else {
            throw new Error(data.detail || 'Gagal mengunggah dokumen.');
        }
    } catch (error) {
        uploadStatus.querySelector('.text').textContent = 'Upload Gagal';
        uploadStatus.querySelector('.dot').style.backgroundColor = 'var(--danger)';
        alert(error.message);
    } finally {
        uploadBtn.disabled = false;
        // Reset input file agar bisa upload file yang sama lagi jika perlu
        fileUpload.value = '';
        
        // Sembunyikan status setelah 3 detik
        setTimeout(() => {
            uploadStatus.style.display = 'none';
        }, 3000);
    }
});
