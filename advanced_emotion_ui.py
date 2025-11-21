import cv2
import numpy as np
import os
import time
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings
warnings.filterwarnings('ignore')

class AdvancedEmotionUI:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)
        
        # مدل و داده‌ها
        self.model = None
        self.is_trained = False
        self.training_data = []
        self.training_labels = []
        
        # آمار و تاریخچه
        self.emotion_history = []
        self.session_start_time = time.time()
        self.faces_detected = 0
        self.predictions_made = 0
        
        # احساسات
        self.emotions = {
            0: {"name": "😠 عصبانی", "color": (0, 0, 255), "count": 0},
            1: {"name": "😄 شاد", "color": (0, 255, 0), "count": 0},
            2: {"name": "😢 غمگین", "color": (255, 0, 0), "count": 0},
            3: {"name": "😲 متعجب", "color": (0, 255, 255), "count": 0},
            4: {"name": "😐 خنثی", "color": (255, 255, 0), "count": 0}
        }
        
        # تنظیمات UI
        self.ui_scale = 1.0
        self.show_charts = True
        self.dark_mode = True
        self.current_view = "main"  # main, stats, training
        
        # ایجاد پوشه‌ها
        self.setup_folders()
    
    def setup_folders(self):
        """ایجاد پوشه‌های لازم"""
        folders = ['sessions', 'exports', 'screenshots']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def create_chart_image(self):
        """ایجاد نمودار آمار احساسات"""
        try:
            # محاسبه آمار
            emotion_counts = {emotion: 0 for emotion in self.emotions}
            for emotion_id in self.emotion_history[-50:]:  # آخرین ۵۰ تشخیص
                if emotion_id in emotion_counts:
                    emotion_counts[emotion_id] += 1
            
            # ایجاد نمودار
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # نمودار میله‌ای
            emotions_list = [self.emotions[i]["name"] for i in range(5)]
            counts = [emotion_counts[i] for i in range(5)]
            colors = [self.emotions[i]["color"] for i in range(5)]
            
            # تبدیل BGR به RGB برای matplotlib
            colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
            
            bars = ax1.bar(emotions_list, counts, color=colors_rgb, alpha=0.7)
            ax1.set_title('توزیع احساسات (آخرین ۵۰ تشخیص)')
            ax1.tick_params(axis='x', rotation=45)
            
            # نمودار دایره‌ای
            total = sum(counts)
            if total > 0:
                ax2.pie(counts, labels=emotions_list, colors=colors_rgb, autopct='%1.1f%%')
                ax2.set_title('درصد احساسات')
            
            plt.tight_layout()
            
            # تبدیل نمودار به تصویر OpenCV
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            chart_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            chart_image = chart_image.reshape(canvas.get_width_height()[::-1] + (3,))
            chart_image = cv2.cvtColor(chart_image, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            return chart_image
            
        except Exception as e:
            print(f"❌ خطا در ایجاد نمودار: {e}")
            return None
    
    def draw_modern_ui(self, frame, faces, current_emotion=None, confidence=0.0):
        """رسم رابط کاربری مدرن"""
        height, width = frame.shape[:2]
        
        # پس‌زمینه نیمه شفاف برای اطلاعات
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # هدر
        header_text = "🤖 سیستم پیشرفته تشخیص احساسات"
        cv2.putText(frame, header_text, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # آمار لحظه‌ای
        runtime = int(time.time() - self.session_start_time)
        stats_text = f"⏱️ {runtime}s | 👥 {self.faces_detected} | 🎯 {self.predictions_made}"
        cv2.putText(frame, stats_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # وضعیت مدل
        model_status = "✅ ML فعال" if self.is_trained else "⚠️ مدل پایه"
        cv2.putText(frame, model_status, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.is_trained else (0, 165, 255), 1)
        
        # نمایش احساس فعلی
        if current_emotion is not None and len(faces) > 0:
            emotion_data = self.emotions[current_emotion]
            emotion_display = f"{emotion_data['name']} | اطمینان: {confidence:.0%}"
            cv2.putText(frame, emotion_display, (width - 400, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_data["color"], 2)
        
        # نوار وضعیت پایین
        cv2.rectangle(frame, (0, height-40), (width, height), (0, 0, 0), -1)
        
        # راهنمای کلیدها
        help_items = [
            "F1: اصلی", "F2: آمار", "F3: آموزش", 
            "S: عکس", "C: نمودار", "Q: خروج"
        ]
        
        x_pos = 20
        for item in help_items:
            cv2.putText(frame, item, (x_pos, height-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            x_pos += 120
    
    def draw_stats_view(self, frame):
        """نمایش نمای آمار"""
        height, width = frame.shape[:2]
        
        # پس‌زمینه
        frame.fill(50)  # خاکستری تیره
        
        # عنوان
        title = "📊 آمار و نمودارها"
        cv2.putText(frame, title, (width//2 - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # ایجاد و نمایش نمودار
        chart = self.create_chart_image()
        if chart is not None:
            # تغییر سایز نمودار برای نمایش
            chart_resized = cv2.resize(chart, (width-100, height-150))
            y_offset = 80
            frame[y_offset:y_offset+chart_resized.shape[0], 
                  50:50+chart_resized.shape[1]] = chart_resized
        
        # آمار کلی
        stats_y = height - 60
        total_detections = len(self.emotion_history)
        if total_detections > 0:
            dominant_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
            dominant_name = self.emotions[dominant_emotion]["name"]
            stats_text = f"🔍 کل تشخیص‌ها: {total_detections} | 🏆 احساس غالب: {dominant_name}"
            cv2.putText(frame, stats_text, (50, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def draw_training_view(self, frame):
        """نمایش نمای آموزش"""
        height, width = frame.shape[:2]
        
        # پس‌زمینه
        frame.fill(60)
        
        # عنوان
        title = "🎓 بخش آموزش مدل"
        cv2.putText(frame, title, (width//2 - 120, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # اطلاعات آموزش
        training_info = [
            f"📊 نمونه‌های آموزشی: {len(self.training_data)}",
            f"🤖 وضعیت مدل: {'آموزش دیده' if self.is_trained else 'نیاز به آموزش'}",
            f"🎯 حداقل نمونه برای آموزش: ۱۰ نمونه از هر احساس"
        ]
        
        y_pos = 100
        for info in training_info:
            cv2.putText(frame, info, (100, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
        
        # راهنمای برچسب‌گذاری
        y_pos += 20
        cv2.putText(frame, "🎯 برچسب‌گذاری احساسات:", (100, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y_pos += 25
        
        emotion_guides = [
            "0: 😠 عصبانی", "1: 😄 شاد", "2: 😢 غمگین", 
            "3: 😲 متعجب", "4: 😐 خنثی"
        ]
        
        x_pos = 100
        for guide in emotion_guides:
            cv2.putText(frame, guide, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            x_pos += 150
        
        # دکمه‌های مجازی
        y_pos += 50
        buttons = [
            ("T: آموزش مدل", (100, y_pos)),
            ("L: بارگذاری مدل", (250, y_pos)),
            ("S: ذخیره داده", (400, y_pos))
        ]
        
        for text, pos in buttons:
            cv2.rectangle(frame, (pos[0]-10, pos[1]-20), (pos[0]+120, pos[1]+5), (100, 100, 100), -1)
            cv2.putText(frame, text, pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_screenshot(self, frame):
        """ذخیره عکس از صفحه"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 عکس ذخیره شد: {filename}")
    
    def export_session_data(self):
        """خروجی گرفتن از داده‌های جلسه"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/session_{timestamp}.json"
        
        session_data = {
            "session_date": datetime.now().isoformat(),
            "duration": int(time.time() - self.session_start_time),
            "total_faces": self.faces_detected,
            "total_predictions": self.predictions_made,
            "emotion_distribution": {self.emotions[i]["name"]: self.emotions[i]["count"] for i in range(5)},
            "emotion_history": self.emotion_history
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            print(f"💾 داده‌های جلسه ذخیره شد: {filename}")
        except Exception as e:
            print(f"❌ خطا در ذخیره داده‌ها: {e}")
    
    def predict_emotion_basic(self, face_roi):
        """پیش‌بینی احساسات (روش پایه)"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness > 170:
                return 1, 0.8  # شاد
            elif brightness < 100:
                return 2, 0.7  # غمگین
            elif brightness > 200:
                return 3, 0.75  # متعجب
            else:
                return 4, 0.6  # خنثی
        except:
            return 4, 0.5
    
    def run(self):
        """اجرای برنامه اصلی"""
        if self.face_cascade.empty() or not self.cap.isOpened():
            print("❌ خطا در راه‌اندازی سیستم")
            return
        
        print("🎨 سیستم تشخیص احساسات با رابط کاربری پیشرفته")
        print("=" * 60)
        print("✅ سیستم آماده است")
        print("\n🎯 راهنمای کلیدها:")
        print("F1: نمای اصلی | F2: آمار | F3: آموزش")
        print("S: عکس از صفحه | E: خروجی داده | Q: خروج")
        print("0-4: برچسب‌گذاری احساسات")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # تشخیص چهره (فقط در نمای اصلی)
            faces = []
            current_emotion = None
            confidence = 0.0
            
            if self.current_view == "main":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
                
                if len(faces) > 0:
                    self.faces_detected += len(faces)
                    
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]
                        emotion_id, conf = self.predict_emotion_basic(face_roi)
                        current_emotion = emotion_id
                        confidence = conf
                        
                        # به روزرسانی آمار
                        self.emotions[emotion_id]["count"] += 1
                        self.emotion_history.append(emotion_id)
                        self.predictions_made += 1
                        
                        # رسم مستطیل دور چهره
                        emotion_data = self.emotions[emotion_id]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_data["color"], 3)
                        
                        # نمایش اطلاعات چهره
                        emotion_text = f"{emotion_data['name']}"
                        cv2.putText(frame, emotion_text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_data["color"], 2)
            
            # رسم UI بر اساس نمای فعلی
            if self.current_view == "main":
                self.draw_modern_ui(frame, faces, current_emotion, confidence)
            elif self.current_view == "stats":
                self.draw_stats_view(frame)
            elif self.current_view == "training":
                self.draw_training_view(frame)
            
            # نمایش پنجره
            window_title = "🎭 تشخیص احساسات - "
            if self.current_view == "main":
                window_title += "نمای اصلی"
            elif self.current_view == "stats":
                window_title += "آمار و نمودارها"
            elif self.current_view == "training":
                window_title += "آموزش مدل"
            
            cv2.imshow(window_title, frame)
            
            # مدیریت کلیدها
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == 0:  # F1
                self.current_view = "main"
            elif key == 0:  # F2
                self.current_view = "stats"
            elif key == 0:  # F3
                self.current_view = "training"
            elif key == ord('s') or key == ord('S'):
                self.save_screenshot(frame)
            elif key == ord('e') or key == ord('E'):
                self.export_session_data()
            elif ord('0') <= key <= ord('4'):
                emotion_id = key - ord('0')
                emotion_name = self.emotions[emotion_id]["name"]
                print(f"🏷️  حالت برچسب‌گذاری: {emotion_name}")
        
        # پایان برنامه
        self.cap.release()
        cv2.destroyAllWindows()
        
        # نمایش آمار نهایی
        runtime = int(time.time() - self.session_start_time)
        print(f"\n📊 آمار نهایی جلسه:")
        print(f"⏱️  مدت زمان: {runtime} ثانیه")
        print(f"👥 چهره‌های تشخیص داده شده: {self.faces_detected}")
        print(f"🎯 پیش‌بینی‌های انجام شده: {self.predictions_made}")
        
        if self.emotion_history:
            dominant = max(set(self.emotion_history), key=self.emotion_history.count)
            print(f"🏆 احساس غالب: {self.emotions[dominant]['name']}")

if __name__ == "__main__":
    ui_system = AdvancedEmotionUI()
    ui_system.run()