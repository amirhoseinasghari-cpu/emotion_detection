from flask import Flask, render_template_string, request, jsonify, send_file
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

class EmotionWebApp:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.analysis_history = []
        self.setup_folders()
    
    def setup_folders(self):
        """ایجاد پوشه‌های مورد نیاز"""
        folders = ['web_results', 'web_exports', 'web_charts']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def process_image(self, image_data):
        """پردازش تصویر و تشخیص احساسات"""
        try:
            # تبدیل base64 به تصویر
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # تشخیص چهره
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            results = []
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotion, confidence = self.detect_emotion(face_roi)
                
                # ذخیره در تاریخچه
                analysis_data = {
                    "timestamp": datetime.now().isoformat(),
                    "emotion": emotion["name"],
                    "emoji": emotion["emoji"],
                    "confidence": float(confidence),
                    "face_location": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                }
                self.analysis_history.append(analysis_data)
                
                # محدود کردن تاریخچه
                if len(self.analysis_history) > 50:
                    self.analysis_history.pop(0)
                
                results.append({
                    "emotion": emotion["name"],
                    "emoji": emotion["emoji"],
                    "confidence": round(confidence, 3),
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y), 
                        "width": int(w),
                        "height": int(h)
                    },
                    "color": emotion["color"]
                })
                
                # رسم روی تصویر
                cv2.rectangle(frame, (x, y), (x+w, y+h), emotion["color"], 3)
                text = f"{emotion['emoji']} {emotion['name']} ({confidence:.0%})"
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion["color"], 2)
            
            # تبدیل فریم پردازش شده به base64
            _, buffer = cv2.imencode('.jpg', frame)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "success": True,
                "faces_detected": len(faces),
                "analysis": results,
                "processed_image": f"data:image/jpeg;base64,{processed_image}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "faces_detected": 0,
                "analysis": []
            }
    
    def detect_emotion(self, face_roi):
        """تشخیص احساسات پیشرفته"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # تقسیم چهره به نواحی
            top_half = gray[0:height//2, :]  # ناحیه چشم‌ها
            bottom_half = gray[height//2:, :]  # ناحیه دهان
            
            # محاسبه ویژگی‌ها
            brightness = np.mean(gray)
            contrast = np.std(gray)
            top_bottom_ratio = np.mean(top_half) / max(np.mean(bottom_half), 1)
            
            # احساسات
            emotions = [
                {"name": "عصبانی", "emoji": "😠", "color": (0, 0, 255), "threshold": (80, 60)},
                {"name": "شاد", "emoji": "😄", "color": (0, 255, 0), "threshold": (170, 70)},
                {"name": "غمگین", "emoji": "😢", "color": (255, 0, 0), "threshold": (90, 50)},
                {"name": "متعجب", "emoji": "😲", "color": (0, 255, 255), "threshold": (200, 80)},
                {"name": "خنثی", "emoji": "😐", "color": (255, 255, 0), "threshold": (150, 40)},
                {"name": "مشوش", "emoji": "😵", "color": (255, 0, 255), "threshold": (120, 30)}
            ]
            
            # منطق تشخیص
            for emotion in emotions:
                brightness_thresh, contrast_thresh = emotion["threshold"]
                if brightness > brightness_thresh and contrast > contrast_thresh:
                    confidence = min((brightness + contrast) / 400, 0.95)
                    return emotion, confidence
            
            # حالت پیش‌فرض
            return emotions[4], 0.7  # خنثی
            
        except Exception as e:
            default_emotion = {"name": "خطا", "emoji": "❌", "color": (255, 255, 255)}
            return default_emotion, 0.0
    
    def get_stats(self):
        """دریافت آمار سیستم"""
        if not self.analysis_history:
            return {"total_analyses": 0, "emotion_distribution": {}}
        
        emotion_counts = {}
        total_confidence = 0
        
        for analysis in self.analysis_history:
            emotion = analysis["emotion"]
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
            total_confidence += analysis["confidence"]
        
        avg_confidence = total_confidence / len(self.analysis_history)
        
        return {
            "total_analyses": len(self.analysis_history),
            "average_confidence": round(avg_confidence, 3),
            "emotion_distribution": emotion_counts
        }
    
    def generate_chart(self):
        """ایجاد نمودار آمار"""
        if not self.analysis_history:
            return None
        
        try:
            # ایجاد نمودار
            plt.figure(figsize=(10, 6))
            
            df = pd.DataFrame(self.analysis_history)
            emotion_counts = df['emotion'].value_counts()
            
            colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', '#FF44FF', '#888888']
            bars = plt.bar(emotion_counts.index, emotion_counts.values, 
                          color=colors[:len(emotion_counts)], alpha=0.7)
            
            plt.title('توزیع احساسات تشخیص داده شده', fontsize=14, fontweight='bold')
            plt.xlabel('احساسات', fontsize=12)
            plt.ylabel('تعداد', fontsize=12)
            plt.xticks(rotation=45)
            
            # اضافه کردن اعداد
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # ذخیره نمودار
            chart_buffer = BytesIO()
            plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            chart_buffer.seek(0)
            chart_data = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{chart_data}"
            
        except Exception as e:
            print(f"خطا در ایجاد نمودار: {e}")
            return None
    
    def export_data(self, format_type='json'):
        """خروجی گرفتن از داده‌ها"""
        if not self.analysis_history:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == 'json':
                filename = f"web_exports/emotion_data_{timestamp}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_history, f, ensure_ascii=False, indent=2)
                return filename
                
            elif format_type == 'excel':
                filename = f"web_exports/emotion_data_{timestamp}.xlsx"
                df = pd.DataFrame(self.analysis_history)
                df.to_excel(filename, index=False)
                return filename
                
        except Exception as e:
            print(f"خطا در خروجی گرفتن: {e}")
            return None

# ایجاد نمونه اپ
emotion_app = EmotionWebApp()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>اپلیکیشن پیشرفته تشخیص احساسات</title>
    <style>
        :root {
            --primary: #4facfe;
            --secondary: #00f2fe;
            --success: #4CAF50;
            --danger: #F44336;
            --warning: #FF9800;
            --dark: #333;
            --light: #f8f9fa;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        
        .camera-section {
            background: var(--light);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        #video {
            width: 100%;
            max-width: 500px;
            border-radius: 10px;
            border: 3px solid var(--primary);
            background: #000;
        }
        
        #processedImage {
            width: 100%;
            max-width: 500px;
            border-radius: 10px;
            border: 3px solid var(--success);
            display: none;
        }
        
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 20px 0;
        }
        
        button {
            padding: 15px 20px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-section {
            background: var(--light);
            padding: 20px;
            border-radius: 15px;
        }
        
        .emotion-display {
            font-size: 4em;
            text-align: center;
            margin: 20px 0;
            height: 80px;
        }
        
        .confidence-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            transition: width 0.5s ease;
            border-radius: 10px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid var(--primary);
        }
        
        .stat-number {
            font-size: 1.5em;
            font-weight: bold;
            color: var(--primary);
        }
        
        .history {
            margin-top: 30px;
        }
        
        .history-item {
            background: white;
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 8px;
            border-right: 4px solid var(--primary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .history-emoji {
            font-size: 1.5em;
        }
        
        .export-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 20px;
        }
        
        .chart-section {
            grid-column: 1 / -1;
            background: var(--light);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        #chartImage {
            max-width: 100%;
            border-radius: 10px;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 اپلیکیشن تشخیص احساسات</h1>
            <p>صورت خود را جلوی دوربین بگیرید و احساساتتان را تحلیل کنید</p>
        </div>
        
        <div class="content">
            <div class="camera-section">
                <h3>📷 دوربین</h3>
                <video id="video" autoplay playsinline></video>
                <img id="processedImage" alt="تصویر پردازش شده">
                <canvas id="canvas" style="display: none;"></canvas>
                
                <div class="controls">
                    <button id="captureBtn" onclick="captureAndAnalyze()">📸 تحلیل احساسات</button>
                    <button id="toggleCameraBtn" onclick="toggleCamera()">🔄 تغییر دوربین</button>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>در حال پردازش...</p>
                </div>
            </div>
            
            <div class="result-section">
                <h3>🎯 نتایج تحلیل</h3>
                <div class="emotion-display" id="emotionDisplay">😐</div>
                <div id="emotionText" style="text-align: center; font-size: 1.2em; margin: 10px 0;">
                    منتظر تحلیل...
                </div>
                
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceBar" style="width: 0%"></div>
                </div>
                <div id="confidenceText" style="text-align: center;">اطمینان: 0%</div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number" id="totalAnalyses">0</div>
                        <div>تعداد تحلیل‌ها</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="facesDetected">0</div>
                        <div>چهره‌های تشخیص داده شده</div>
                    </div>
                </div>
                
                <div class="export-buttons">
                    <button onclick="exportData('json')">💾 ذخیره JSON</button>
                    <button onclick="exportData('excel')">📊 ذخیره Excel</button>
                </div>
            </div>
            
            <div class="chart-section">
                <h3>📊 نمودار آمار</h3>
                <img id="chartImage" src="" alt="نمودار آمار">
                <button onclick="updateChart()" style="margin-top: 15px;">🔄 به روزرسانی نمودار</button>
            </div>
            
            <div class="history">
                <h3>📝 تاریخچه تحلیل‌ها</h3>
                <div id="historyList"></div>
            </div>
        </div>
    </div>

    <script>
        // متغیرهای全局
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let context = canvas.getContext('2d');
        let currentStream = null;
        let isFrontCamera = true;
        
        // راه‌اندازی دوربین
        async function setupCamera() {
            try {
                const constraints = {
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: isFrontCamera ? 'user' : 'environment'
                    } 
                };
                
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                
                if (currentStream) {
                    currentStream.getTracks().forEach(track => track.stop());
                }
                
                currentStream = stream;
                video.srcObject = stream;
                
                return new Promise((resolve) => {
                    video.onloadedmetadata = () => {
                        resolve(video);
                    };
                });
            } catch (err) {
                alert('❌ خطا در دسترسی به دوربین: ' + err.message);
                return null;
            }
        }
        
        // تغییر دوربین
        async function toggleCamera() {
            isFrontCamera = !isFrontCamera;
            await setupCamera();
        }
        
        // گرفتن عکس و تحلیل
        async function captureAndAnalyze() {
            const loading = document.getElementById('loading');
            const captureBtn = document.getElementById('captureBtn');
            
            loading.style.display = 'block';
            captureBtn.disabled = true;
            
            try {
                const imageData = captureImage();
                const result = await analyzeImage(imageData);
                displayResult(result);
                updateStats();
                updateHistory();
                updateChart();
            } catch (error) {
                alert('❌ خطا: ' + error.message);
            } finally {
                loading.style.display = 'none';
                captureBtn.disabled = false;
            }
        }
        
        // گرفتن عکس از دوربین
        function captureImage() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);
            return canvas.toDataURL('image/jpeg');
        }
        
        // تحلیل تصویر
        async function analyzeImage(imageData) {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) {
                throw new Error('خطا در ارتباط با سرور');
            }
            
            return await response.json();
        }
        
        // نمایش نتیجه
        function displayResult(result) {
            const emotionDisplay = document.getElementById('emotionDisplay');
            const emotionText = document.getElementById('emotionText');
            const confidenceBar = document.getElementById('confidenceBar');
            const confidenceText = document.getElementById('confidenceText');
            const processedImage = document.getElementById('processedImage');
            const facesDetected = document.getElementById('facesDetected');
            
            facesDetected.textContent = result.faces_detected;
            
            if (result.success && result.analysis.length > 0) {
                const analysis = result.analysis[0];
                emotionDisplay.textContent = analysis.emoji;
                emotionText.textContent = `احساس: ${analysis.emotion}`;
                confidenceBar.style.width = (analysis.confidence * 100) + '%';
                confidenceText.textContent = `اطمینان: ${(analysis.confidence * 100).toFixed(1)}%`;
                
                // نمایش تصویر پردازش شده
                processedImage.src = result.processed_image;
                processedImage.style.display = 'block';
                video.style.display = 'none';
                
                // بازگشت به دوربین بعد از 3 ثانیه
                setTimeout(() => {
                    processedImage.style.display = 'none';
                    video.style.display = 'block';
                }, 3000);
                
            } else {
                emotionDisplay.textContent = '❌';
                emotionText.textContent = 'چهره‌ای تشخیص داده نشد';
                confidenceBar.style.width = '0%';
                confidenceText.textContent = 'اطمینان: 0%';
            }
        }
        
        // به روزرسانی آمار
        async function updateStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                
                document.getElementById('totalAnalyses').textContent = stats.total_analyses;
            } catch (error) {
                console.error('خطا در دریافت آمار:', error);
            }
        }
        
        // به روزرسانی تاریخچه
        async function updateHistory() {
            try {
                const response = await fetch('/history');
                const history = await response.json();
                
                const historyList = document.getElementById('historyList');
                historyList.innerHTML = '';
                
                history.slice(-10).reverse().forEach(item => {
                    const historyItem = document.createElement('div');
                    historyItem.className = 'history-item';
                    historyItem.innerHTML = `
                        <div>
                            <strong>${item.emoji} ${item.emotion}</strong>
                            <div style="font-size: 0.8em; color: #666;">
                                ${new Date(item.timestamp).toLocaleString('fa-IR')}
                            </div>
                        </div>
                        <div>${(item.confidence * 100).toFixed(1)}%</div>
                    `;
                    historyList.appendChild(historyItem);
                });
            } catch (error) {
                console.error('خطا در دریافت تاریخچه:', error);
            }
        }
        
        // به روزرسانی نمودار
        async function updateChart() {
            try {
                const response = await fetch('/chart');
                const chartData = await response.text();
                
                if (chartData) {
                    document.getElementById('chartImage').src = chartData;
                }
            } catch (error) {
                console.error('خطا در دریافت نمودار:', error);
            }
        }
        
        // خروجی گرفتن از داده‌ها
        async function exportData(format) {
            try {
                const response = await fetch(`/export/${format}`);
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = `emotion_data.${format}`;
                    link.click();
                    
                    alert('✅ داده‌ها با موفقیت ذخیره شدند');
                } else {
                    alert('❌ خطا در ذخیره داده‌ها');
                }
            } catch (error) {
                alert('❌ خطا: ' + error.message);
            }
        }
        
        // راه‌اندازی اولیه
        document.addEventListener('DOMContentLoaded', async function() {
            await setupCamera();
            await updateStats();
            await updateHistory();
            await updateChart();
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return HTML_TEMPLATE

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    result = emotion_app.process_image(data['image'])
    return jsonify(result)

@app.route('/stats')
def stats():
    return jsonify(emotion_app.get_stats())

@app.route('/history')
def history():
    return jsonify(emotion_app.analysis_history[-20:])  # آخرین 20 مورد

@app.route('/chart')
def chart():
    chart_data = emotion_app.generate_chart()
    if chart_data:
        return chart_data
    else:
        return ""

@app.route('/export/<format_type>')
def export_data(format_type):
    filename = emotion_app.export_data(format_type)
    if filename:
        return send_file(filename, as_attachment=True)
    else:
        return jsonify({"error": "No data to export"}), 400

if __name__ == '__main__':
    print("🚀 اپلیکیشن تحت وب پیشرفته راه‌اندازی شد!")
    print("📍 آدرس: http://localhost:5000")
    print("📱 قابل استفاده روی موبایل و دسکتاپ")
    app.run(host='0.0.0.0', port=5000, debug=True)