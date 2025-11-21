// إعدادات API
const API_URL = 'http://localhost:5000';

// عناصر DOM
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraBtn = document.getElementById('startCamera');
const stopCameraBtn = document.getElementById('stopCamera');
const registerModeBtn = document.getElementById('registerMode');
const recognizeModeBtn = document.getElementById('recognizeMode');
const registerForm = document.getElementById('registerForm');
const recognizeForm = document.getElementById('recognizeForm');
const captureRegisterBtn = document.getElementById('captureRegister');
const captureRecognizeBtn = document.getElementById('captureRecognize');
const userNameInput = document.getElementById('userName');
const resultDiv = document.getElementById('result');
const usersList = document.getElementById('usersList');
const userCount = document.getElementById('userCount');
const refreshUsersBtn = document.getElementById('refreshUsers');

// متغيرات
let stream = null;
let currentMode = 'register'; // 'register' or 'recognize'

// تهيئة التطبيق
document.addEventListener('DOMContentLoaded', () => {
    loadUsers();
    setupEventListeners();
});

// إعداد Event Listeners
function setupEventListeners() {
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);
    registerModeBtn.addEventListener('click', () => switchMode('register'));
    recognizeModeBtn.addEventListener('click', () => switchMode('recognize'));
    captureRegisterBtn.addEventListener('click', captureAndRegister);
    captureRecognizeBtn.addEventListener('click', captureAndRecognize);
    refreshUsersBtn.addEventListener('click', loadUsers);
}

// تبديل الوضع بين التسجيل والتعرف
function switchMode(mode) {
    currentMode = mode;
    
    if (mode === 'register') {
        registerModeBtn.classList.add('active');
        recognizeModeBtn.classList.remove('active');
        registerForm.style.display = 'block';
        recognizeForm.style.display = 'none';
    } else {
        registerModeBtn.classList.remove('active');
        recognizeModeBtn.classList.add('active');
        registerForm.style.display = 'none';
        recognizeForm.style.display = 'block';
    }
    
    hideResult();
}

// تشغيل الكاميرا
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            } 
        });
        
        video.srcObject = stream;
        video.play();
        
        startCameraBtn.style.display = 'none';
        stopCameraBtn.style.display = 'inline-flex';
        
        showResult('تم تشغيل الكاميرا بنجاح', 'success');
    } catch (error) {
        console.error('Error accessing camera:', error);
        showResult('فشل في الوصول إلى الكاميرا. تأكد من السماح بالوصول', 'error');
    }
}

// إيقاف الكاميرا
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
        
        startCameraBtn.style.display = 'inline-flex';
        stopCameraBtn.style.display = 'none';
        
        showResult('تم إيقاف الكاميرا', 'info');
    }
}

// التقاط صورة من الفيديو
function captureImage() {
    if (!stream) {
        showResult('يرجى تشغيل الكاميرا أولاً', 'error');
        return null;
    }
    
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    return canvas.toDataURL('image/jpeg', 0.9);
}

// إظهار الرسائل العامة (تختفي بعد 6 ثواني)
function showResult(message, type) {
    resultDiv.textContent = message;
    resultDiv.className = `result ${type}`;
    resultDiv.style.display = 'block';
    setTimeout(() => hideResult(), 6000);
}

// إظهار الرسائل الخاصة بالعد التنازلي أو تعليمات التقاط الصور الثلاث
function showCountdownMessage(message) {
    resultDiv.textContent = message;
    resultDiv.className = `result info`;
    resultDiv.style.display = 'block';
}

// التقاط والتسجيل (3 صور مع عد تنازلي)
async function captureAndRegister() {
    const name = userNameInput.value.trim();
    
    if (!name) {
        showResult('يرجى إدخال الاسم', 'error');
        userNameInput.focus();
        return;
    }
    
    captureRegisterBtn.disabled = true;

    const imagesData = [];
    const totalImages = 3;
    
    for (let i = 0; i < totalImages; i++) {
        showCountdownMessage(`استعد للصورة ${i + 1} من ${totalImages}`);
        
        // انتظار 2 ثانية قبل التقاط الصورة
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const imageData = captureImage();
        if (!imageData) {
            showResult('فشل في التقاط الصورة', 'error');
            captureRegisterBtn.disabled = false;
            return;
        }
        
        imagesData.push(imageData);
    }

    showResult('جاري تسجيل البيانات...', 'info');

    try {
        const response = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name, images: imagesData })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult(`✅ ${data.message}`, 'success');
            userNameInput.value = '';
            loadUsers();
        } else {
            showResult(`❌ ${data.message}`, 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showResult('❌ فشل الاتصال بالسيرفر', 'error');
    } finally {
        captureRegisterBtn.disabled = false;
    }
}

// التقاط والتعرف
async function captureAndRecognize() {
    const imageData = captureImage();
    if (!imageData) return;
    
    captureRecognizeBtn.disabled = true;
    showResult('جاري التعرف على الوجه...', 'info');
    
    try {
        const response = await fetch(`${API_URL}/recognize`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.recognized) {
                const confidence = (data.confidence * 100).toFixed(1);
                showResult(`✅ مرحباً ${data.name}! (دقة التعرف: ${confidence}%)`, 'success');
            } else {
                showResult('❌ الوجه غير معروف. يرجى التسجيل أولاً', 'error');
            }
        } else {
            showResult(`❌ ${data.message}`, 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showResult('❌ فشل الاتصال بالسيرفر', 'error');
    } finally {
        captureRecognizeBtn.disabled = false;
    }
}

// تحميل قائمة المستخدمين
async function loadUsers() {
    try {
        const response = await fetch(`${API_URL}/users`);
        const data = await response.json();
        if (data.success) {
            displayUsers(data.users);
            userCount.textContent = data.count;
        }
    } catch (error) {
        console.error('Error loading users:', error);
        usersList.innerHTML = '<div class="empty-state"><span>⚠️</span><p>فشل تحميل المستخدمين</p></div>';
    }
}

// عرض المستخدمين
function displayUsers(users) {
    if (users.length === 0) {
        usersList.innerHTML = '<div class="empty-state"><span>👤</span><p>لا يوجد مستخدمين مسجلين</p></div>';
        return;
    }
    usersList.innerHTML = users.map(user => `
        <div class="user-card">
            <div class="user-info">
                <span class="user-icon">👤</span>
                <span class="user-name">${user}</span>
            </div>
            <button class="delete-btn" onclick="deleteUser('${user}')">🗑️</button>
        </div>
    `).join('');
}

// حذف مستخدم
async function deleteUser(name) {
    if (!confirm(`هل أنت متأكد من حذف ${name}؟`)) return;
    
    try {
        const response = await fetch(`${API_URL}/delete/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const data = await response.json();
        
        if (data.success) {
            showResult(`✅ ${data.message}`, 'success');
            loadUsers();
        } else {
            showResult(`❌ ${data.message}`, 'error');
        }
    } catch (error) {
        console.error('Error deleting user:', error);
        showResult('❌ فشل حذف المستخدم', 'error');
    }
}

// إخفاء النتيجة
function hideResult() {
    resultDiv.style.display = 'none';
}

// التنظيف عند إغلاق الصفحة
window.addEventListener('beforeunload', () => {
    stopCamera();
});
