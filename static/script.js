// Global utility functions and shared functionality

// Navigation function
function navigateTo(page) {
    window.location.href = page;
}

// Language management
function changeLanguage(lang = null) {
    if (lang) {
        localStorage.setItem('app_language', lang);
    } else {
        lang = document.getElementById('languageSelect')?.value || localStorage.getItem('app_language') || 'en';
        localStorage.setItem('app_language', lang);
    }
    
    // Update all elements with language attributes
    document.querySelectorAll('[data-en]').forEach(element => {
        const text = element.getAttribute(`data-${lang}`);
        if (text) {
            element.textContent = text;
        }
    });
    
    // Update placeholders
    document.querySelectorAll('[data-placeholder-en]').forEach(element => {
        const placeholder = element.getAttribute(`data-placeholder-${lang}`);
        if (placeholder) {
            element.placeholder = placeholder;
        }
    });
}

// User management
function getCurrentUser() {
    const userStr = localStorage.getItem('current_user');
    return userStr ? JSON.parse(userStr) : null;
}

function setCurrentUser(user) {
    localStorage.setItem('current_user', JSON.stringify(user));
}

// Authentication functions
function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const currentLang = localStorage.getItem('app_language') || 'en';
    
    if (!email || !password) {
        const fillFieldsText = currentLang === 'hi' ? 'कृपया सभी फ़ील्ड भरें' : 'Please fill in all fields';
        showNotification(fillFieldsText, 'error');
        return;
    }
    
    // Simulate login process
    const user = {
        id: Date.now(),
        email: email,
        name: email.split('@')[0],
        role: getUrlParameter('role') || 'patient',
        loginTime: new Date().toISOString()
    };
    
    setCurrentUser(user);
    const successText = currentLang === 'hi' ? 'लॉगिन सफल!' : 'Login successful!';
    showNotification(successText);
    
    setTimeout(() => {
        navigateTo('dashboard.html');
    }, 1000);
}

function handleSignup(event) {
    event.preventDefault();
    
    const name = document.getElementById('signupName').value;
    const email = document.getElementById('signupEmail').value;
    const phone = document.getElementById('signupPhone').value;
    const password = document.getElementById('signupPassword').value;
    const dueDate = document.getElementById('dueDate').value;
    const currentLang = localStorage.getItem('app_language') || 'en';
    
    if (!name || !email || !phone || !password) {
        const fillFieldsText = currentLang === 'hi' ? 'कृपया सभी आवश्यक फ़ील्ड भरें' : 'Please fill in all required fields';
        showNotification(fillFieldsText, 'error');
        return;
    }
    
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        const emailErrorText = currentLang === 'hi' ? 'कृपया एक वैध ईमेल पता दर्ज करें' : 'Please enter a valid email address';
        showNotification(emailErrorText, 'error');
        return;
    }
    
    // Validate phone format (basic)
    const phoneRegex = /^[0-9]{10}$/;
    if (!phoneRegex.test(phone.replace(/\D/g, ''))) {
        const phoneErrorText = currentLang === 'hi' ? 'कृपया एक वैध 10-अंकीय फोन नंबर दर्ज करें' : 'Please enter a valid 10-digit phone number';
        showNotification(phoneErrorText, 'error');
        return;
    }
    
    // Create user account
    const user = {
        id: Date.now(),
        name: name,
        email: email,
        phone: phone,
        role: getUrlParameter('role') || 'patient',
        signupTime: new Date().toISOString()
    };
    
    // Save user profile
    const profile = {
        name: name,
        email: email,
        phone: phone,
        createdAt: new Date().toISOString()
    };
    
    localStorage.setItem('user_profile', JSON.stringify(profile));
    
    // Save due date if provided
    if (dueDate) {
        localStorage.setItem('pregnancy_due_date', new Date(dueDate).toISOString());
    }
    
    setCurrentUser(user);
    const successText = currentLang === 'hi' ? 'खाता सफलतापूर्वक बनाया गया!' : 'Account created successfully!';
    showNotification(successText);
    
    setTimeout(() => {
        navigateTo('dashboard.html');
    }, 1000);
}

// Tab functionality
function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Notification system
function showNotification(message, type = 'success') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());
    
    // Create new notification
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 3000);
}

// URL parameter helper
function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

// Local storage helpers for reports
function saveReportToStorage(report) {
    const reports = getAllReports();
    reports.push(report);
    localStorage.setItem('health_reports', JSON.stringify(reports));
}

function getAllReports() {
    const reportsStr = localStorage.getItem('health_reports');
    return reportsStr ? JSON.parse(reportsStr) : [];
}

function deleteReportFromStorage(reportId) {
    let reports = getAllReports();
    reports = reports.filter(report => report.id != reportId);
    localStorage.setItem('health_reports', JSON.stringify(reports));
}

// Date and time utilities
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function formatTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getRelativeTime(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);
    
    if (diffInSeconds < 60) {
        return 'Just now';
    } else if (diffInSeconds < 3600) {
        const minutes = Math.floor(diffInSeconds / 60);
        return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    } else if (diffInSeconds < 86400) {
        const hours = Math.floor(diffInSeconds / 3600);
        return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
        const days = Math.floor(diffInSeconds / 86400);
        return `${days} day${days > 1 ? 's' : ''} ago`;
    }
}

// Health data utilities
function calculateBMI(weight, height) {
    // weight in kg, height in cm
    const heightInMeters = height / 100;
    return (weight / (heightInMeters * heightInMeters)).toFixed(1);
}

function getBMICategory(bmi) {
    if (bmi < 18.5) return 'Underweight';
    if (bmi < 25) return 'Normal';
    if (bmi < 30) return 'Overweight';
    return 'Obese';
}

function calculatePregnancyWeek(dueDate) {
    const due = new Date(dueDate);
    const now = new Date();
    const timeDiff = due.getTime() - now.getTime();
    const daysDiff = Math.ceil(timeDiff / (1000 * 3600 * 24));
    
    // Pregnancy is typically 280 days (40 weeks)
    const totalDays = 280;
    const daysPassed = totalDays - daysDiff;
    return Math.max(1, Math.min(42, Math.ceil(daysPassed / 7)));
}

// Data validation utilities
function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function validatePhone(phone) {
    const phoneRegex = /^[0-9]{10}$/;
    return phoneRegex.test(phone.replace(/\D/g, ''));
}

function validateDate(dateString) {
    const date = new Date(dateString);
    return date instanceof Date && !isNaN(date);
}

// Image handling utilities
function resizeImage(file, maxWidth = 800, maxHeight = 600, quality = 0.8) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = function() {
            // Calculate new dimensions
            let { width, height } = img;
            
            if (width > height) {
                if (width > maxWidth) {
                    height = (height * maxWidth) / width;
                    width = maxWidth;
                }
            } else {
                if (height > maxHeight) {
                    width = (width * maxHeight) / height;
                    height = maxHeight;
                }
            }
            
            // Resize image
            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);
            
            // Convert to blob
            canvas.toBlob(resolve, 'image/jpeg', quality);
        };
        
        img.src = URL.createObjectURL(file);
    });
}

// Export utilities
function exportToCSV(data, filename) {
    const csvContent = "data:text/csv;charset=utf-8," + data;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function exportToJSON(data, filename) {
    const jsonContent = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
    const link = document.createElement("a");
    link.setAttribute("href", jsonContent);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Reminder and notification utilities
function scheduleReminder(time, message, type = 'iron') {
    if ('Notification' in window && Notification.permission === 'granted') {
        const now = new Date();
        const reminderTime = new Date();
        const [hours, minutes] = time.split(':');
        reminderTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);
        
        // If time has passed today, schedule for tomorrow
        if (reminderTime <= now) {
            reminderTime.setDate(reminderTime.getDate() + 1);
        }
        
        const timeUntilReminder = reminderTime.getTime() - now.getTime();
        
        setTimeout(() => {
            new Notification('AI-nemia Reminder', {
                body: message,
                icon: '/favicon.ico',
                badge: '/favicon.ico'
            });
        }, timeUntilReminder);
    }
}

function requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

// Emergency contact utilities
function callEmergencyNumber(number) {
    const currentLang = localStorage.getItem('app_language') || 'en';
    const callText = currentLang === 'hi' ? `${number} पर कॉल करें?` : `Call ${number}?`;
    
    if (confirm(callText)) {
        window.location.href = `tel:${number}`;
    }
}

function sendEmergencySMS(number, message) {
    const smsUrl = `sms:${number}?body=${encodeURIComponent(message)}`;
    window.location.href = smsUrl;
}

// Geolocation utilities
function getCurrentLocation() {
    return new Promise((resolve, reject) => {
        if ('geolocation' in navigator) {
            navigator.geolocation.getCurrentPosition(resolve, reject);
        } else {
            reject(new Error('Geolocation not supported'));
        }
    });
}

// Offline detection
function isOnline() {
    return navigator.onLine;
}

function handleOfflineMode() {
    if (!isOnline()) {
        const currentLang = localStorage.getItem('app_language') || 'en';
        const offlineText = currentLang === 'hi' ? 
            'आप ऑफ़लाइन हैं। कुछ सुविधाएं ठीक से काम नहीं कर सकती हैं।' :
            'You are offline. Some features may not work properly.';
        showNotification(offlineText, 'warning');
    }
}

// Initialize offline detection
window.addEventListener('online', () => {
    const currentLang = localStorage.getItem('app_language') || 'en';
    const onlineText = currentLang === 'hi' ? 'कनेक्शन बहाल हो गया!' : 'Connection restored!';
    showNotification(onlineText);
});

window.addEventListener('offline', () => {
    const currentLang = localStorage.getItem('app_language') || 'en';
    const offlineText = currentLang === 'hi' ? 'आप अब ऑफ़लाइन हैं' : 'You are now offline';
    showNotification(offlineText, 'warning');
});

// Performance monitoring
function measurePerformance(name, fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    console.log(`${name} took ${end - start} milliseconds`);
    return result;
}

// Error handling
function handleError(error, context = 'Unknown') {
    console.error(`Error in ${context}:`, error);
    const currentLang = localStorage.getItem('app_language') || 'en';
    const errorText = currentLang === 'hi' ? 
        'एक त्रुटि हुई। कृपया पुनः प्रयास करें।' :
        'An error occurred. Please try again.';
    showNotification(errorText, 'error');
}

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    // Set initial language
    const savedLanguage = localStorage.getItem('app_language') || 'en';
    const languageSelect = document.getElementById('languageSelect');
    if (languageSelect) {
        languageSelect.value = savedLanguage;
        changeLanguage(savedLanguage);
    }
    
    // Request notification permission
    requestNotificationPermission();
    
    // Check online status
    handleOfflineMode();
    
    // Initialize any page-specific functionality
    const currentPage = window.location.pathname.split('/').pop();
    initializePage(currentPage);
});

function initializePage(page) {
    switch (page) {
        case 'index.html':
        case '':
            // Splash screen initialization
            break;
        case 'dashboard.html':
            // Dashboard initialization
            break;
        case 'detection.html':
            // Detection page initialization
            break;
        case 'vitals.html':
            // Vitals page initialization
            break;
        case 'nutrition.html':
            // Nutrition page initialization
            setupIronReminders();
            break;
        case 'symptoms.html':
            // Symptoms page initialization
            break;
        case 'reports.html':
            // Reports page initialization
            break;
        case 'pregnancy.html':
            // Pregnancy page initialization
            break;
        case 'schemes.html':
            // Schemes page initialization
            break;
        case 'profile.html':
            // Profile page initialization
            break;
    }
}

function setupIronReminders() {
    const reminderEnabled = localStorage.getItem('iron_reminder_enabled') === 'true';
    const reminderTime = localStorage.getItem('iron_reminder_time') || '09:00';
    const currentLang = localStorage.getItem('app_language') || 'en';
    
    if (reminderEnabled) {
        const reminderText = currentLang === 'hi' ? 
            'आयरन टैबलेट लेने का समय!' :
            'Time to take your iron tablet!';
        scheduleReminder(reminderTime, reminderText, 'iron');
    }
}

// Service Worker registration for offline support
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}

// PWA install prompt
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    
    // Show install button or banner
    showInstallPrompt();
});

function showInstallPrompt() {
    const currentLang = localStorage.getItem('app_language') || 'en';
    const installText = currentLang === 'hi' ? 
        'त्वरित पहुंच और ऑफ़लाइन उपयोग के लिए AI-नेमिया इंस्टॉल करें' :
        'Install AI-nemia for quick access and offline use';
    const installBtnText = currentLang === 'hi' ? 'इंस्टॉल करें' : 'Install';
    
    // Create install prompt UI
    const installBanner = document.createElement('div');
    installBanner.className = 'install-banner';
    installBanner.innerHTML = `
        <div class="install-content">
            <p>${installText}</p>
            <button onclick="installApp()" class="install-btn">${installBtnText}</button>
            <button onclick="dismissInstall()" class="dismiss-btn">×</button>
        </div>
    `;
    
    document.body.appendChild(installBanner);
}

function installApp() {
    if (deferredPrompt) {
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('User accepted the install prompt');
            }
            deferredPrompt = null;
        });
    }
    dismissInstall();
}

function dismissInstall() {
    const installBanner = document.querySelector('.install-banner');
    if (installBanner) {
        installBanner.remove();
    }
}

// Analytics and usage tracking (privacy-friendly)
function trackPageView(page) {
    const usage = JSON.parse(localStorage.getItem('app_usage') || '{}');
    usage[page] = (usage[page] || 0) + 1;
    usage.lastVisit = new Date().toISOString();
    localStorage.setItem('app_usage', JSON.stringify(usage));
}

function trackFeatureUsage(feature) {
    const features = JSON.parse(localStorage.getItem('feature_usage') || '{}');
    features[feature] = (features[feature] || 0) + 1;
    localStorage.setItem('feature_usage', JSON.stringify(features));
}

// Accessibility helpers
function announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 1000);
}

// Keyboard navigation helpers
function handleKeyboardNavigation(event) {
    if (event.key === 'Escape') {
        // Close any open modals
        const openModals = document.querySelectorAll('.modal[style*="flex"]');
        openModals.forEach(modal => {
            modal.style.display = 'none';
        });
    }
}

document.addEventListener('keydown', handleKeyboardNavigation);

// Touch and gesture support
let touchStartX = 0;
let touchStartY = 0;

document.addEventListener('touchstart', function(e) {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
});

document.addEventListener('touchend', function(e) {
    if (!touchStartX || !touchStartY) return;
    
    const touchEndX = e.changedTouches[0].clientX;
    const touchEndY = e.changedTouches[0].clientY;
    
    
    const diffX = touchStartX - touchEndX;
    const diffY = touchStartY - touchEndY;
    
    // Detect swipe gestures
    if (Math.abs(diffX) > Math.abs(diffY)) {
        if (Math.abs(diffX) > 50) {
            if (diffX > 0) {
                // Swipe left
                handleSwipeLeft();
            } else {
                // Swipe right
                handleSwipeRight();
            }
        }
    }
    
    touchStartX = 0;
    touchStartY = 0;
});

function handleSwipeLeft() {
    // Handle left swipe (e.g., next page)
}

function handleSwipeRight() {
    // Handle right swipe (e.g., back/previous page)
    const backBtn = document.querySelector('.back-btn');
    if (backBtn) {
        backBtn.click();
    }
}

// Data backup and restore
function backupData() {
    const data = {
        profile: localStorage.getItem('user_profile'),
        reports: localStorage.getItem('health_reports'),
        dueDate: localStorage.getItem('pregnancy_due_date'),
        settings: {
            language: localStorage.getItem('app_language'),
            notifications: localStorage.getItem('notifications_enabled'),
            ironReminder: localStorage.getItem('iron_reminder_enabled')
        },
        timestamp: new Date().toISOString()
    };
    
    return JSON.stringify(data);
}

function restoreData(backupString) {
    try {
        const data = JSON.parse(backupString);
        
        if (data.profile) localStorage.setItem('user_profile', data.profile);
        if (data.reports) localStorage.setItem('health_reports', data.reports);
        if (data.dueDate) localStorage.setItem('pregnancy_due_date', data.dueDate);
        if (data.settings) {
            Object.entries(data.settings).forEach(([key, value]) => {
                if (value !== null) localStorage.setItem(key, value);
            });
        }
        
        const currentLang = localStorage.getItem('app_language') || 'en';
        const restoredText = currentLang === 'hi' ? 'डेटा सफलतापूर्वक पुनर्स्थापित किया गया!' : 'Data restored successfully!';
        showNotification(restoredText);
        return true;
    } catch (error) {
        const currentLang = localStorage.getItem('app_language') || 'en';
        const errorText = currentLang === 'hi' ? 
            'डेटा पुनर्स्थापित करने में विफल। अमान्य बैकअप फ़ाइल।' :
            'Failed to restore data. Invalid backup file.';
        showNotification(errorText, 'error');
        return false;
    }
}

// Memory management
function clearUnusedData() {
    // Clear old temporary data
    const keys = Object.keys(localStorage);
    keys.forEach(key => {
        if (key.startsWith('temp_') || key.startsWith('cache_')) {
            const item = localStorage.getItem(key);
            try {
                const data = JSON.parse(item);
                if (data.timestamp) {
                    const age = Date.now() - new Date(data.timestamp).getTime();
                    if (age > 24 * 60 * 60 * 1000) { // 24 hours
                        localStorage.removeItem(key);
                    }
                }
            } catch (e) {
                // Remove invalid items
                localStorage.removeItem(key);
            }
        }
    });
}

// Run cleanup on app start
clearUnusedData();

// Auto-save functionality
let autoSaveTimeout;

function scheduleAutoSave(data, key) {
    clearTimeout(autoSaveTimeout);
    autoSaveTimeout = setTimeout(() => {
        localStorage.setItem(key, JSON.stringify(data));
    }, 1000); // Save after 1 second of inactivity
}

// Initialize page tracking
trackPageView(window.location.pathname);