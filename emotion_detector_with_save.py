import cv2
import numpy as np
import os
import time
import json
from datetime import datetime

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)
        self.emotions = {
            0: {"name": "😠 عصبانی", "color": (0, 0, 255)},
            1: {"name": "😄 شاد", "color": (0, 255, 0)},
            2: {"name": "😢 غمگین", "color": (255, 0, 0)},
            3: {"name": "😲 متعجب", "color": (0, 255, 255)},
            4: {"name": "😐 خنثی", "color": (255, 255, 0)}
        }
        
        # ایجاد پوشه‌های ذخیره‌سازی
        self.setup_folders()
        
        # داده‌های جمع‌آوری شده
        self.collected_data = []
        self.photo_count = 0
    
    def setup_folders(self):
        """ایجاد پوشه‌های لازم برای ذخیره‌سازی"""
        folders = ['saved_faces', 'data_logs']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"✅ پوشه {folder}/ ایجاد شد")
    
    def save_face_data(self, face_roi, emotion_id, confidence):
        """ذخیره عکس چهره و اطلاعات مربوطه"""
        try:
            # تولید نام فایل
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"saved_faces/face_{timestamp}_{emotion_id}_{self.photo_count}.jpg"
            
            # ذخیره عکس
            cv2.imwrite(filename, face_roi)
            
            # ذخیره اطلاعات متا
            face_data = {
                "filename": filename,
                "emotion_id": emotion_id,
                "emotion_name": self.emotions[emotion_id]["name"],
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat(),
                "face_size": f"{face_roi.shape[1]}x{face_roi.shape[0]}"
            }
            
            self.collected_data.append(face_data)
            self.photo_count += 1
            
            print(f"📸 ذخیره شد: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ خطا در ذخیره‌سازی: {e}")
            return False
    
    def save_data_log(self):
        """ذخیره کلیه داده‌ها در فایل JSON"""
        if not self.collected_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"data_logs/emotion_data_{timestamp}.json"
        
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
            
            print(f"📊 داده‌ها ذخیره شد: {log_filename}")
            print(f"📈 تعداد رکوردها: {len(self.collected_data)}")
            
        except Exception as e:
            print(f"❌ خطا در ذخیره لاگ: {e}")
    
    def show_statistics(self):
        """نمایش آمار داده‌های جمع‌آوری شده"""
        if not self.collected_data:
            print("📭 هیچ داده‌ای جمع‌آوری نشده است")
            return
        
        emotion_counts = {}
        total_confidences = {}
        
        for data in self.collected_data:
            emotion_name = data["emotion_name"]
            if emotion_name not in emotion_counts:
                emotion_counts[emotion_name] = 0
                total_confidences[emotion_name] = 0
            emotion_counts[emotion_name] += 1
            total_confidences[emotion_name] += data["confidence"]
        
        print("\n📊 آمار داده‌های جمع‌آوری شده:")
        print("=" * 40)
        for emotion, count in emotion_counts.items():
            avg_confidence = total_confidences[emotion] / count
            print(f"{emotion}: {count} نمونه (میانگین اطمینان: {avg_confidence:.1%})")
    
    def analyze_face_features(self, face_roi):
        """آنالیز ویژگی‌های چهره برای تشخیص احساسات"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            top_half = gray[0:height//2, :]
            bottom_half = gray[height//2:, :]
            
            brightness = np.mean(gray)
            contrast = np.std(gray)
            top_bottom_ratio = np.mean(top_half) / max(np.mean(bottom_half), 1)
            
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
                
        except Exception as e:
            return 4, 0.5
    
    def run(self):
        """اجرای برنامه اصلی"""
        if self.face_cascade.empty() or not self.cap.isOpened():
            print("❌ خطا در راه‌اندازی سیستم")
            return
        
        print("🎭 سیستم تشخیص احساسات با ذخیره‌سازی داده")
        print("=" * 50)
        print("✅ سیستم آماده است")
        print("\n🎯 راهنما:")
        print("- A: ذخیره خودکار عکس‌ها (هر 3 ثانیه)")
        print("- S: ذخیره دستی عکس فعلی")
        print("- D: نمایش آمار داده‌ها")
        print("- Q: خروج و ذخیره نهایی")
        
        start_time = time.time()
        detection_history = []
        auto_save = False
        last_auto_save = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # تشخیص چهره
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            current_emotion = None
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotion_id, confidence = self.analyze_face_features(face_roi)
                emotion_data = self.emotions[emotion_id]
                current_emotion = emotion_id
                
                # ذخیره در تاریخچه
                detection_history.append(emotion_id)
                if len(detection_history) > 20:
                    detection_history.pop(0)
                
                # رسم مستطیل با رنگ احساس
                cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_data["color"], 3)
                
                # نمایش احساسات
                emotion_text = f"{emotion_data['name']} ({confidence:.0%})"
                cv2.putText(frame, emotion_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_data["color"], 2)
                
                # ذخیره‌سازی خودکار
                if auto_save and time.time() - last_auto_save > 3:
                    self.save_face_data(face_roi, emotion_id, confidence)
                    last_auto_save = time.time()
            
            # نمایش اطلاعات
            runtime = int(time.time() - start_time)
            stats_text = f"چهره‌ها: {len(faces)} | زمان: {runtime}s | عکس‌ها: {self.photo_count}"
            cv2.putText(frame, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # وضعیت ذخیره‌سازی خودکار
            auto_status = "ذخیره خودکار: فعال" if auto_save else "ذخیره خودکار: غیرفعال"
            cv2.putText(frame, auto_status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if auto_save else (0, 0, 255), 2)
            
            # راهنمای کلیدها
            help_text = "A:ذخیره خودکار S:ذخیره دستی D:آمار Q:خروج"
            cv2.putText(frame, help_text, (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("تشخیص احساسات - ذخیره داده", frame)
            
            # مدیریت کلیدها
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break
            elif key == ord('a') or key == ord('A'):
                auto_save = not auto_save
                status = "فعال" if auto_save else "غیرفعال"
                print(f"🔧 ذخیره خودکار {status} شد")
            elif key == ord('s') or key == ord('S'):
                if faces and current_emotion is not None:
                    face_roi = frame[y:y+h, x:x+w]
                    emotion_id, confidence = self.analyze_face_features(face_roi)
                    self.save_face_data(face_roi, emotion_id, confidence)
            elif key == ord('d') or key == ord('D'):
                self.show_statistics()
        
        # ذخیره نهایی و تمیز کردن
        self.save_data_log()
        self.show_statistics()
        self.cap.release()
        cv2.destroyAllWindows()
        
        runtime = int(time.time() - start_time)
        print(f"\n✅ برنامه پایان یافت!")
        print(f"⏱️  مدت اجرا: {runtime} ثانیه")
        print(f"📸 تعداد عکس‌های ذخیره شده: {self.photo_count}")

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run()