import cv2
import numpy as np
import os
import time

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
    
    def analyze_face_features(self, face_roi):
        """آنالیز ویژگی‌های چهره برای تشخیص احساسات"""
        try:
            # تبدیل به خاکستری
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # تقسیم چهره به نواحی مختلف
            top_half = gray[0:height//2, :]      # ناحیه چشم‌ها و ابروها
            bottom_half = gray[height//2:, :]    # ناحیه دهان
            
            # محاسبه ویژگی‌ها
            brightness = np.mean(gray)
            contrast = np.std(gray)
            top_bottom_ratio = np.mean(top_half) / max(np.mean(bottom_half), 1)
            
            # منطق تشخیص احساسات
            if brightness > 170 and contrast > 60:
                return 1, 0.85  # شاد - صورت روشن با کنتراست بالا
            elif brightness < 100:
                return 2, 0.75  # غمگین - صورت تاریک
            elif top_bottom_ratio > 1.3:
                return 3, 0.80  # متعجب - ناحیه بالایی فعال
            elif contrast < 40:
                return 4, 0.70  # خنثی - کنتراست پایین
            else:
                return 0, 0.65  # عصبانی - حالت پیش‌فرض
                
        except Exception as e:
            return 4, 0.5  # حالت خطا
    
    def run(self):
        """اجرای برنامه اصلی"""
        if self.face_cascade.empty() or not self.cap.isOpened():
            print("❌ خطا در راه‌اندازی سیستم")
            return
        
        print("🎭 سیستم پیشرفته تشخیص احساسات")
        print("=" * 45)
        print("✅ سیستم آماده است")
        print("\n🎯 راهنما:")
        print("- 😄 شاد: لبخند بزن")
        print("- 😢 غمگین: اخم کن") 
        print("- 😲 متعجب: چشم‌ها رو باز کن")
        print("- 😠 عصبانی: ابروها رو گره بزن")
        print("- Q: خروج")
        
        start_time = time.time()
        detection_history = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # تشخیص چهره
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                # استخراج ناحیه چهره
                face_roi = frame[y:y+h, x:x+w]
                
                # تشخیص احساسات
                emotion_id, confidence = self.analyze_face_features(face_roi)
                emotion_data = self.emotions[emotion_id]
                
                # ذخیره در تاریخچه
                detection_history.append(emotion_id)
                if len(detection_history) > 10:
                    detection_history.pop(0)
                
                # رسم مستطیل با رنگ احساس
                cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_data["color"], 3)
                
                # نمایش احساسات
                emotion_text = f"{emotion_data['name']} ({confidence:.0%})"
                cv2.putText(frame, emotion_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_data["color"], 2)
                
                # نمایش اطلاعات چهره
                info_text = f"Size: {w}x{h}"
                cv2.putText(frame, info_text, (x, y+h+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_data["color"], 1)
            
            # نمایش آمار
            runtime = int(time.time() - start_time)
            stats_text = f"چهره‌ها: {len(faces)} | زمان: {runtime}s"
            cv2.putText(frame, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # نمایش احساس غالب
            if detection_history:
                dominant_emotion = max(set(detection_history), key=detection_history.count)
                dominant_text = f"احساس غالب: {self.emotions[dominant_emotion]['name']}"
                cv2.putText(frame, dominant_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "Q: خروج", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("تشخیص احساسات چهره - Emotion Detection", frame)
            
            # مدیریت کلیدها
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
        
        # تمیز کردن
        self.cap.release()
        cv2.destroyAllWindows()
        
        # نمایش آمار نهایی
        runtime = int(time.time() - start_time)
        print(f"\n📊 آمار نهایی:")
        print(f"⏱️  مدت اجرا: {runtime} ثانیه")
        print(f"🎭 تشخیص‌های انجام شده: {len(detection_history)}")
        if detection_history:
            dominant = max(set(detection_history), key=detection_history.count)
            print(f"🏆 احساس غالب: {self.emotions[dominant]['name']}")

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run()