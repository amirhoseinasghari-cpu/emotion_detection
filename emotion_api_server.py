from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import time
import threading
from datetime import datetime
import os

app = Flask(__name__)

class EmotionAPI:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.analysis_history = []
        self.request_count = 0
        self.start_time = time.time()
        
        # احساسات
        self.emotions = {
            0: {"name": "عصبانی", "emoji": "😠", "color": [255, 0, 0]},
            1: {"name": "شاد", "emoji": "😄", "color": [0, 255, 0]},
            2: {"name": "غمگین", "emoji": "😢", "color": [0, 0, 255]},
            3: {"name": "متعجب", "emoji": "😲", "color": [255, 255, 0]},
            4: {"name": "خنثی", "emoji": "😐", "color": [128, 128, 128]}
        }
        
        # ایجاد پوشه برای لاگ
        if not os.path.exists('api_logs'):
            os.makedirs('api_logs')
    
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
                emotion_id, confidence = self.analyze_emotion(face_roi)
                emotion_data = self.emotions[emotion_id]
                
                # ذخیره در تاریخچه
                analysis_data = {
                    "timestamp": datetime.now().isoformat(),
                    "emotion": emotion_data["name"],
                    "confidence": confidence,
                    "face_location": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                }
                self.analysis_history.append(analysis_data)
                
                # محدود کردن تاریخچه به 1000 رکورد
                if len(self.analysis_history) > 1000:
                    self.analysis_history.pop(0)
                
                results.append({
                    "emotion": emotion_data["name"],
                    "emoji": emotion_data["emoji"],
                    "confidence": round(confidence, 3),
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y), 
                        "width": int(w),
                        "height": int(h)
                    },
                    "color": emotion_data["color"]
                })
            
            self.request_count += 1
            
            return {
                "success": True,
                "faces_detected": len(faces),
                "analysis": results,
                "processing_time": time.time() - self.start_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "faces_detected": 0,
                "analysis": []
            }
    
    def analyze_emotion(self, face_roi):
        """آنالیز احساسات چهره"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # تقسیم چهره به نواحی
            top_half = gray[0:height//2, :]
            bottom_half = gray[height//2:, :]
            
            # محاسبه ویژگی‌ها
            brightness = np.mean(gray)
            contrast = np.std(gray)
            top_bottom_ratio = np.mean(top_half) / max(np.mean(bottom_half), 1)
            
            # منطق تشخیص
            if brightness > 170 and contrast > 60:
                return 1, 0.85  # شاد
            elif brightness < 100:
                return 2, 0.75  # غمگین
            elif top_bottom_ratio > 1.3:
                return 3, 0.80  # متعجب
            elif contrast < 40:
                return 4, 0.70  # خنثی
            else:
                return 0, 0.65  # عصبانی
                
        except:
            return 4, 0.5
    
    def get_stats(self):
        """دریافت آمار سیستم"""
        emotion_counts = {}
        total_confidence = 0
        total_analyses = len(self.analysis_history)
        
        for analysis in self.analysis_history[-100:]:  # آخرین 100 تحلیل
            emotion = analysis["emotion"]
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
            total_confidence += analysis["confidence"]
        
        avg_confidence = total_confidence / max(total_analyses, 1)
        
        return {
            "total_requests": self.request_count,
            "total_analyses": total_analyses,
            "uptime": round(time.time() - self.start_time, 2),
            "average_confidence": round(avg_confidence, 3),
            "recent_emotion_distribution": emotion_counts
        }
    
    def save_logs(self):
        """ذخیره لاگ‌ها"""
        if not self.analysis_history:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"api_logs/emotion_logs_{timestamp}.json"
        
        try:
            log_data = {
                "export_time": datetime.now().isoformat(),
                "total_requests": self.request_count,
                "analyses": self.analysis_history[-1000:]  # آخرین 1000 رکورد
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 لاگ‌ها ذخیره شد: {log_file}")
            
        except Exception as e:
            print(f"❌ خطا در ذخیره لاگ: {e}")

# ایجاد نمونه API
emotion_api = EmotionAPI()

# Routes
@app.route('/')
def home():
    """صفحه اصلی"""
    return """
    <html>
        <head>
            <title>API تشخیص احساسات چهره</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
                code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
                .demo-section { margin: 30px 0; padding: 20px; background: #e7f3ff; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 API تشخیص احساسات چهره</h1>
                <p>سرویس REST API برای تشخیص خودکار احساسات از تصاویر چهره</p>
                
                <div class="endpoint">
                    <h3>📤 POST /analyze</h3>
                    <p>آنالیز تصویر و تشخیص احساسات</p>
                    <p><strong>پارامتر:</strong> <code>{"image": "base64_image_data"}</code></p>
                </div>
                
                <div class="endpoint">
                    <h3>📊 GET /stats</h3>
                    <p>دریافت آمار و وضعیت سیستم</p>
                </div>
                
                <div class="endpoint">
                    <h3>🔄 GET /health</h3>
                    <p>بررسی سلامت سرویس</p>
                </div>
                
                <div class="demo-section">
                    <h3>🎯 تست آنلاین API</h3>
                    <input type="file" id="imageInput" accept="image/*">
                    <button onclick="analyzeImage()">آنالیز تصویر</button>
                    <div id="result"></div>
                </div>
                
                <script>
                    function analyzeImage() {
                        const input = document.getElementById('imageInput');
                        const resultDiv = document.getElementById('result');
                        
                        if (!input.files[0]) {
                            resultDiv.innerHTML = '<p style="color: red;">لطفا یک تصویر انتخاب کنید</p>';
                            return;
                        }
                        
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            fetch('/analyze', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({image: e.target.result})
                            })
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    let html = `<h4>نتایج آنالیز:</h4>`;
                                    html += `<p>تعداد چهره‌ها: ${data.faces_detected}</p>`;
                                    data.analysis.forEach((face, index) => {
                                        html += `<div style="margin: 10px 0; padding: 10px; border-left: 4px solid rgb(${face.color.join(',')})">
                                            <strong>${face.emoji} ${face.emotion}</strong> (اطمینان: ${(face.confidence * 100).toFixed(1)}%)
                                        </div>`;
                                    });
                                    resultDiv.innerHTML = html;
                                } else {
                                    resultDiv.innerHTML = `<p style="color: red;">خطا: ${data.error}</p>`;
                                }
                            })
                            .catch(error => {
                                resultDiv.innerHTML = `<p style="color: red;">خطا در ارتباط با سرور</p>`;
                            });
                        };
                        reader.readAsDataURL(input.files[0]);
                    }
                </script>
            </div>
        </body>
    </html>
    """

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """آنالیز تصویر و تشخیص احساسات"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "پارامتر image الزامی است"
            }), 400
        
        result = emotion_api.process_image(data['image'])
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """دریافت آمار سیستم"""
    stats = emotion_api.get_stats()
    return jsonify({
        "success": True,
        "stats": stats,
        "server_time": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """بررسی سلامت سرویس"""
    return jsonify({
        "status": "healthy",
        "service": "Emotion Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/logs/save', methods=['POST'])
def save_logs():
    """ذخیره لاگ‌های سیستم"""
    emotion_api.save_logs()
    return jsonify({
        "success": True,
        "message": "لاگ‌ها با موفقیت ذخیره شدند"
    })

# Scheduled log saving (هر 10 دقیقه)
def scheduled_log_saving():
    while True:
        time.sleep(600)  # 10 دقیقه
        emotion_api.save_logs()

# راه‌اندازی ذخیره‌سازی زمان‌بندی شده
log_thread = threading.Thread(target=scheduled_log_saving, daemon=True)
log_thread.start()

if __name__ == '__main__':
    print("🌐 راه‌اندازی API تشخیص احساسات چهره...")
    print("📍 آدرس سرور: http://localhost:5000")
    print("📚 endpoints در دسترس:")
    print("   GET  /          - مستندات و تست آنلاین")
    print("   POST /analyze   - آنالیز تصویر")
    print("   GET  /stats     - آمار سیستم") 
    print("   GET  /health    - سلامت سرویس")
    print("   POST /logs/save - ذخیره لاگ‌ها")
    print("\n🚀 سرور در حال اجرا...")
    
    app.run(host='0.0.0.0', port=5000, debug=True)