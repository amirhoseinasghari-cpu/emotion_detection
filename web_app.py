from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
from datetime import datetime
import os

app = Flask(__name__)

class WebEmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.analysis_history = []
        
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
                self.analysis_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "emotion": emotion["name"],
                    "confidence": confidence,
                    "face_location": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                })
                
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
            
            # تبدیل فریم به base64 برای نمایش
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
        """تشخیص احساسات"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            emotions = [
                {"name": "عصبانی", "emoji": "😠", "color": [255, 0, 0], "threshold": 80},
                {"name": "شاد", "emoji": "😄", "color": [0, 255, 0], "threshold": 170},
                {"name": "غمگین", "emoji": "😢", "color": [0, 0, 255], "threshold": 90},
                {"name": "متعجب", "emoji": "😲", "color": [255, 255, 0], "threshold": 200},
                {"name": "خنثی", "emoji": "😐", "color": [128, 128, 128], "threshold": 150}
            ]
            
            for emotion in emotions:
                if brightness > emotion["threshold"]:
                    confidence = min(brightness / 255, 0.95)
                    return emotion, confidence
                    
            return emotions[4], 0.7  # حالت پیش‌فرض
            
        except:
            default_emotion = {"name": "خطا", "emoji": "❌", "color": [255, 255, 255]}
            return default_emotion, 0.0

# ایجاد نمونه تشخیص‌گر
detector = WebEmotionDetector()

@app.route('/')
def index():
    """صفحه اصلی اپ تحت وب"""
    return """
    <!DOCTYPE html>
    <html lang="fa">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>اپلیکیشن تشخیص احساسات</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 30px;
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
            }
            
            .camera-section {
                text-align: center;
                margin-bottom: 30px;
            }
            
            #video {
                width: 100%;
                max-width: 500px;
                border-radius: 10px;
                border: 3px solid #4facfe;
            }
            
            #canvas {
                display: none;
            }
            
            .controls {
                display: flex;
                gap: 10px;
                justify-content: center;
                margin: 20px 0;
                flex-wrap: wrap;
            }
            
            button {
                padding: 15px 30px;
                border: none;
                border-radius: 50px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                min-width: 150px;
            }
            
            #captureBtn {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
            }
            
            #saveBtn {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #333;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            }
            
            .result {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                text-align: center;
            }
            
            .emotion-display {
                font-size: 3em;
                margin: 10px 0;
            }
            
            .confidence-bar {
                width: 100%;
                height: 20px;
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .confidence-fill {
                height: 100%;
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                transition: width 0.5s ease;
            }
            
            .history {
                margin-top: 30px;
            }
            
            .history-item {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 10px;
                border-left: 5px solid #4facfe;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            @media (max-width: 600px) {
                .header h1 {
                    font-size: 2em;
                }
                
                button {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📱 تشخیص احساسات</h1>
                <p>صورت خود را جلوی دوربین بگیرید و احساساتتان را تحلیل کنید</p>
            </div>
            
            <div class="content">
                <div class="camera-section">
                    <video id="video" autoplay playsinline></video>
                    <canvas id="canvas"></canvas>
                    
                    <div class="controls">
                        <button id="captureBtn">📸 تحلیل احساسات</button>
                        <button id="saveBtn">💾 ذخیره نتیجه</button>
                    </div>
                </div>
                
                <div class="result" id="resultSection" style="display: none;">
                    <h3>نتیجه تحلیل:</h3>
                    <div class="emotion-display" id="emotionDisplay">😐</div>
                    <div id="emotionText">در حال پردازش...</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidenceBar" style="width: 0%"></div>
                    </div>
                    <div id="confidenceText">اطمینان: 0%</div>
                </div>
                
                <div class="history">
                    <h3>📊 تاریخچه تحلیل‌ها</h3>
                    <div id="historyList"></div>
                </div>
            </div>
        </div>

        <script>
            // متغیرهای全局
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let context = canvas.getContext('2d');
            let analysisHistory = [];
            
            // راه‌اندازی دوربین
            async function setupCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: { ideal: 640 },
                            height: { ideal: 480 },
                            facingMode: 'user'
                        } 
                    });
                    video.srcObject = stream;
                    
                    return new Promise((resolve) => {
                        video.onloadedmetadata = () => {
                            resolve(video);
                        };
                    });
                } catch (err) {
                    alert('❌ دسترسی به دوربین ممکن نیست: ' + err.message);
                    return null;
                }
            }
            
            // گرفتن عکس از دوربین
            function captureImage() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0);
                
                return canvas.toDataURL('image/jpeg');
            }
            
            // تحلیل احساسات
            async function analyzeEmotion() {
                const imageData = captureImage();
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ image: imageData })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        displayResult(result);
                    } else {
                        showError(result.error);
                    }
                } catch (error) {
                    showError('خطا در ارتباط با سرور: ' + error.message);
                }
            }
            
            // نمایش نتیجه
            function displayResult(result) {
                const resultSection = document.getElementById('resultSection');
                const emotionDisplay = document.getElementById('emotionDisplay');
                const emotionText = document.getElementById('emotionText');
                const confidenceBar = document.getElementById('confidenceBar');
                const confidenceText = document.getElementById('confidenceText');
                
                resultSection.style.display = 'block';
                
                if (result.analysis.length > 0) {
                    const analysis = result.analysis[0];
                    emotionDisplay.textContent = analysis.emoji;
                    emotionText.textContent = analysis.emotion;
                    confidenceBar.style.width = (analysis.confidence * 100) + '%';
                    confidenceText.textContent = `اطمینان: ${(analysis.confidence * 100).toFixed(1)}%`;
                    
                    // اضافه کردن به تاریخچه
                    addToHistory(analysis);
                } else {
                    emotionDisplay.textContent = '❌';
                    emotionText.textContent = 'چهره‌ای تشخیص داده نشد';
                    confidenceBar.style.width = '0%';
                    confidenceText.textContent = 'اطمینان: 0%';
                }
            }
            
            // اضافه کردن به تاریخچه
            function addToHistory(analysis) {
                analysisHistory.unshift({
                    ...analysis,
                    timestamp: new Date().toLocaleString('fa-IR')
                });
                
                // محدود کردن تاریخچه به 10 مورد
                if (analysisHistory.length > 10) {
                    analysisHistory.pop();
                }
                
                updateHistoryDisplay();
            }
            
            // به روزرسانی نمایش تاریخچه
            function updateHistoryDisplay() {
                const historyList = document.getElementById('historyList');
                historyList.innerHTML = '';
                
                analysisHistory.forEach(item => {
                    const historyItem = document.createElement('div');
                    historyItem.className = 'history-item';
                    historyItem.innerHTML = `
                        <div>
                            <strong>${item.emoji} ${item.emotion}</strong>
                            <div style="font-size: 0.8em; color: #666;">${item.timestamp}</div>
                        </div>
                        <div>${(item.confidence * 100).toFixed(1)}%</div>
                    `;
                    historyList.appendChild(historyItem);
                });
            }
            
            // نمایش خطا
            function showError(message) {
                const resultSection = document.getElementById('resultSection');
                const emotionDisplay = document.getElementById('emotionDisplay');
                const emotionText = document.getElementById('emotionText');
                
                resultSection.style.display = 'block';
                emotionDisplay.textContent = '❌';
                emotionText.textContent = message;
            }
            
            // ذخیره نتایج
            function saveResults() {
                if (analysisHistory.length === 0) {
                    alert('❌ هیچ داده‌ای برای ذخیره وجود ندارد');
                    return;
                }
                
                const dataStr = JSON.stringify(analysisHistory, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                
                const link = document.createElement('a');
                link.href = URL.createObjectURL(dataBlob);
                link.download = `emotion_analysis_${new Date().getTime()}.json`;
                link.click();
                
                alert('✅ نتایج با موفقیت ذخیره شد');
            }
            
            // راه‌اندازی اولیه
            document.addEventListener('DOMContentLoaded', async function() {
                // راه‌اندازی دوربین
                await setupCamera();
                
                // اضافه کردن event listeners
                document.getElementById('captureBtn').addEventListener('click', analyzeEmotion);
                document.getElementById('saveBtn').addEventListener('click', saveResults);
            });
        </script>
    </body>
    </html>
    """

@app.route('/analyze', methods=['POST'])
def analyze():
    """آنالیز تصویر"""
    data = request.get_json()
    result = detector.process_image(data['image'])
    return jsonify(result)

if __name__ == '__main__':
    print("🌐 راه‌اندازی اپلیکیشن تحت وب...")
    print("📍 آدرس: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)