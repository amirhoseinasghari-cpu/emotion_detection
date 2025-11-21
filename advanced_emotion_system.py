import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedEmotionSystem:
    def __init__(self):
        print("🔧 در حال راه‌اندازی سیستم...")
        
        # بارگذاری تشخیص چهره
        if not os.path.exists("haarcascade_frontalface_default.xml"):
            print("❌ فایل تشخیص چهره پیدا نشد!")
            return
        
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # راه‌اندازی وبکم
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ وبکم پیدا نشد!")
            return
        
        # مدل‌های پیشرفته
        self.ml_model = None
        self.is_ml_trained = False
        
        # داده‌های تاریخی
        self.session_data = []
        self.training_data = []
        self.training_labels = []
        
        # احساسات پیشرفته
        self.emotions = {
            0: {"name": "عصبانی", "emoji": "😠", "color": (0, 0, 255), "features": []},
            1: {"name": "شاد", "emoji": "😄", "color": (0, 255, 0), "features": []},
            2: {"name": "غمگین", "emoji": "😢", "color": (255, 0, 0), "features": []},
            3: {"name": "متعجب", "emoji": "😲", "color": (0, 255, 255), "features": []},
            4: {"name": "خنثی", "emoji": "😐", "color": (255, 255, 0), "features": []},
            5: {"name": "مشوش", "emoji": "😵", "color": (255, 0, 255), "features": []}
        }
        
        # آمار پیشرفته
        self.performance_stats = {
            "total_detections": 0,
            "ml_predictions": 0,
            "basic_predictions": 0,
            "average_confidence": 0,
            "start_time": datetime.now()
        }
        
        # تنظیمات سیستم
        self.settings = {
            "use_ml": False,
            "auto_save": True,
            "show_debug": True,
            "detection_interval": 5  # فریم
        }
        
        # ایجاد پوشه‌ها
        self.setup_folders()
        print("✅ سیستم با موفقیت راه‌اندازی شد")
    
    def setup_folders(self):
        """ایجاد پوشه‌های پیشرفته"""
        folders = [
            'advanced_data/raw_faces',
            'advanced_data/training_sets', 
            'advanced_data/models',
            'advanced_reports/analytics',
            'advanced_reports/performance',
            'advanced_exports/datasets'
        ]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"📁 پوشه ایجاد شد: {folder}")
    
    def extract_advanced_features(self, face_roi):
        """استخراج ویژگی‌های پیشرفته از چهره"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # ویژگی‌های پایه
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # ویژگی‌های ساده‌تر برای تست
            features = [brightness, contrast]
            
            return np.array(features)
            
        except Exception as e:
            if self.settings["show_debug"]:
                print(f"⚠️ خطا در استخراج ویژگی‌ها: {e}")
            return None
    
    def analyze_emotion_advanced(self, face_roi):
        """آنالیز احساسات با روش پیشرفته"""
        try:
            features = self.extract_advanced_features(face_roi)
            if features is None:
                return self.analyze_emotion_basic(face_roi)
            
            # استفاده از مدل ML اگر آموزش دیده باشد
            if self.settings["use_ml"] and self.is_ml_trained:
                return self.predict_with_ml(features)
            else:
                return self.analyze_with_features(features)
                
        except Exception as e:
            if self.settings["show_debug"]:
                print(f"⚠️ خطا در آنالیز پیشرفته: {e}")
            return self.analyze_emotion_basic(face_roi)
    
    def analyze_with_features(self, features):
        """آنالیز با ویژگی‌های استخراج شده"""
        brightness = features[0] if len(features) > 0 else 128
        contrast = features[1] if len(features) > 1 else 50
        
        # منطق پیشرفته‌تر
        if brightness > 180 and contrast > 70:
            return 1, 0.88  # شاد
        elif brightness < 80:
            return 2, 0.82  # غمگین
        elif contrast > 100:
            return 3, 0.85  # متعجب
        elif brightness > 150 and contrast < 40:
            return 4, 0.78  # خنثی
        elif contrast < 30:
            return 5, 0.75  # مشوش
        else:
            return 0, 0.80  # عصبانی
    
    def analyze_emotion_basic(self, face_roi):
        """آنالیز احساسات پایه (Fallback)"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            if brightness > 170:
                return 1, 0.8  # شاد
            elif brightness < 90:
                return 2, 0.7  # غمگین
            elif contrast > 80:
                return 3, 0.75  # متعجب
            else:
                return 4, 0.65  # خنثی
        except:
            return 4, 0.5
    
    def collect_training_data(self, face_roi, emotion_id):
        """جمع‌آوری داده برای آموزش ML"""
        features = self.extract_advanced_features(face_roi)
        if features is not None:
            self.training_data.append(features)
            self.training_labels.append(emotion_id)
            
            # ذخیره خودکار
            if self.settings["auto_save"] and len(self.training_data) % 50 == 0:
                self.save_training_dataset()
            
            return True
        return False
    
    def train_ml_model(self):
        """آموزش مدل ماشین لرنینگ پیشرفته"""
        if len(self.training_data) < 10:  # کاهش حداقل نمونه
            print("❌ داده‌های آموزشی کافی نیست (حداقل ۱۰ نمونه)")
            return False
        
        try:
            print("🤖 در حال آموزش مدل پیشرفته...")
            
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42  # افزایش test size
            )
            
            # ایجاد مدل ساده‌تر
            self.ml_model = RandomForestClassifier(
                n_estimators=50,  # کاهش تعداد درختان
                max_depth=10,
                random_state=42
            )
            
            # آموزش
            self.ml_model.fit(X_train, y_train)
            
            # ارزیابی
            y_pred = self.ml_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_ml_trained = True
            self.performance_stats["ml_accuracy"] = accuracy
            
            print(f"✅ مدل پیشرفته آموزش داده شد!")
            print(f"📊 دقت مدل: {accuracy:.1%}")
            print(f"📈 تعداد نمونه‌های آموزشی: {len(self.training_data)}")
            
            # ذخیره مدل
            self.save_ml_model()
            return True
            
        except Exception as e:
            print(f"❌ خطا در آموزش مدل: {e}")
            return False
    
    def predict_with_ml(self, features):
        """پیش‌بینی با مدل ML"""
        try:
            prediction = self.ml_model.predict([features])[0]
            probabilities = self.ml_model.predict_proba([features])[0]
            confidence = probabilities[prediction]
            
            self.performance_stats["ml_predictions"] += 1
            return prediction, confidence
            
        except Exception as e:
            if self.settings["show_debug"]:
                print(f"⚠️ خطا در پیش‌بینی ML: {e}")
            return self.analyze_with_features(features)
    
    def save_ml_model(self):
        """ذخیره مدل آموزش دیده"""
        if self.ml_model is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"advanced_data/models/advanced_model_{timestamp}.pkl"
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.ml_model, f)
            
            print(f"💾 مدل پیشرفته ذخیره شد: {model_path}")
            
        except Exception as e:
            print(f"❌ خطا در ذخیره مدل: {e}")
    
    def save_training_dataset(self):
        """ذخیره مجموعه داده‌های آموزشی"""
        if not self.training_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = f"advanced_data/training_sets/dataset_{timestamp}.npz"
        
        try:
            np.savez_compressed(
                dataset_path,
                data=self.training_data,
                labels=self.training_labels
            )
            
            print(f"💾 مجموعه داده ذخیره شد: {dataset_path}")
            
        except Exception as e:
            print(f"❌ خطا در ذخیره داده‌ها: {e}")
    
    def generate_performance_report(self):
        """ایجاد گزارش عملکرد سیستم"""
        try:
            runtime = datetime.now() - self.performance_stats["start_time"]
            total_predictions = self.performance_stats["ml_predictions"] + self.performance_stats["basic_predictions"]
            
            report = {
                "report_date": datetime.now().isoformat(),
                "runtime_seconds": runtime.total_seconds(),
                "total_detections": self.performance_stats["total_detections"],
                "ml_predictions": self.performance_stats["ml_predictions"],
                "basic_predictions": self.performance_stats["basic_predictions"],
                "ml_usage_ratio": self.performance_stats["ml_predictions"] / max(total_predictions, 1),
                "model_accuracy": self.performance_stats.get("ml_accuracy", 0),
                "training_samples": len(self.training_data),
                "is_ml_trained": self.is_ml_trained
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"advanced_reports/performance/report_{timestamp}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"📈 گزارش عملکرد ایجاد شد: {report_path}")
            return report
            
        except Exception as e:
            print(f"❌ خطا در ایجاد گزارش عملکرد: {e}")
            return None
    
    def draw_advanced_ui(self, frame, faces, current_emotion=None, confidence=0.0):
        """رسم رابط کاربری پیشرفته"""
        height, width = frame.shape[:2]
        
        # پس‌زمینه اطلاعات
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # هدر سیستم
        cv2.putText(frame, "🤖 سیستم پیشرفته تشخیص احساسات", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # وضعیت مدل
        model_status = "ML فعال ✅" if self.settings["use_ml"] and self.is_ml_trained else "ML غیرفعال ⚠️"
        model_color = (0, 255, 0) if self.settings["use_ml"] and self.is_ml_trained else (0, 165, 255)
        cv2.putText(frame, f"وضعیت: {model_status}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, model_color, 1)
        
        # آمار لحظه‌ای
        stats_text = f"تشخیص‌ها: {self.performance_stats['total_detections']} | آموزش: {len(self.training_data)}"
        cv2.putText(frame, stats_text, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # نمایش احساس فعلی (اگر وجود دارد)
        if current_emotion is not None and current_emotion in self.emotions:
            emotion_data = self.emotions[current_emotion]
            emotion_display = f"{emotion_data['emoji']} {emotion_data['name']} | اطمینان: {confidence:.0%}"
            cv2.putText(frame, emotion_display, (width - 400, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_data["color"], 2)
        
        # نوار راهنما
        cv2.rectangle(frame, (0, height-50), (width, height), (0, 0, 0), -1)
        help_text = "M:ML  T:آموزش  S:آمار  D:ذخیره  Q:خروج  0-5:برچسب"
        cv2.putText(frame, help_text, (20, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run_advanced_system(self):
        """اجرای سیستم پیشرفته"""
        # بررسی راه‌اندازی
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            print("❌ سیستم به درستی راه‌اندازی نشده است")
            return
        
        print("🚀 راه‌اندازی سیستم پیشرفته تشخیص احساسات")
        print("=" * 60)
        print("✅ سیستم آماده است")
        print("\n🎯 راهنمای پیشرفته:")
        print("M: فعال/غیرفعال کردن ML")
        print("T: آموزش مدل با داده‌های جمع‌آوری شده") 
        print("S: نمایش آمار و گزارش")
        print("D: ذخیره داده‌ها")
        print("0-5: برچسب‌گذاری احساسات")
        print("Q: خروج")
        print("\n🔍 در حال اجرا...")
        
        frame_count = 0
        current_emotion_label = 4  # پیش‌فرض: خنثی
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ مشکل در دریافت تصویر از وبکم")
                    break
                
                frame_count += 1
                
                # مقداردهی اولیه متغیرها
                current_emotion = None
                current_confidence = 0.0
                faces = []
                
                # تشخیص چهره (با فاصله برای عملکرد بهتر)
                if frame_count % self.settings["detection_interval"] == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
                    
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # آنالیز احساسات
                        emotion_id, confidence = self.analyze_emotion_advanced(face_roi)
                        current_emotion = emotion_id
                        current_confidence = confidence
                        
                        # به روزرسانی آمار
                        self.performance_stats["total_detections"] += 1
                        if not (self.settings["use_ml"] and self.is_ml_trained):
                            self.performance_stats["basic_predictions"] += 1
                        
                        # رسم روی فریم
                        emotion_data = self.emotions[emotion_id]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_data["color"], 3)
                        
                        emotion_text = f"{emotion_data['emoji']} {emotion_data['name']}"
                        cv2.putText(frame, emotion_text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_data["color"], 2)
                        
                        confidence_text = f"{confidence:.0%}"
                        cv2.putText(frame, confidence_text, (x, y+h+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_data["color"], 1)
                
                # رسم UI - حالا current_emotion همیشه تعریف شده
                self.draw_advanced_ui(frame, faces, current_emotion, current_confidence)
                
                cv2.imshow('سیستم پیشرفته تشخیص احساسات - Advanced Emotion System', frame)
                
                # مدیریت کلیدها با تاخیر بیشتر
                key = cv2.waitKey(30) & 0xFF  # افزایش به 30ms
                if key == ord('q') or key == ord('Q'):
                    print("🔴 درخواست خروج...")
                    break
                elif key == ord('m') or key == ord('M'):
                    self.settings["use_ml"] = not self.settings["use_ml"]
                    status = "فعال" if self.settings["use_ml"] else "غیرفعال"
                    print(f"🔧 مدل ML {status} شد")
                elif key == ord('t') or key == ord('T'):
                    self.train_ml_model()
                elif key == ord('s') or key == ord('S'):
                    report = self.generate_performance_report()
                    if report:
                        print("📊 گزارش عملکرد ایجاد شد")
                elif key == ord('d') or key == ord('D'):
                    self.save_training_dataset()
                    self.save_ml_model()
                    print("💾 داده‌ها ذخیره شدند")
                elif ord('0') <= key <= ord('5'):
                    emotion_id = key - ord('0')
                    if emotion_id in self.emotions:
                        current_emotion_label = emotion_id
                        emotion_name = self.emotions[emotion_id]["name"]
                        print(f"🏷️  حالت برچسب‌گذاری: {emotion_name}")
        
        except Exception as e:
            print(f"❌ خطا در اجرای سیستم: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # ذخیره نهایی
            if hasattr(self, 'cap'):
                self.cap.release()
            cv2.destroyAllWindows()
            
            print("\n📦 ذخیره نهایی داده‌ها...")
            if hasattr(self, 'training_data') and self.training_data:
                self.save_training_dataset()
            if hasattr(self, 'performance_stats'):
                self.generate_performance_report()
            
            if hasattr(self, 'performance_stats'):
                runtime = datetime.now() - self.performance_stats["start_time"]
                print(f"\n✅ سیستم پیشرفته پایان یافت!")
                print(f"⏱️  مدت اجرا: {runtime.total_seconds():.1f} ثانیه")
                print(f"📊 تشخیص‌های انجام شده: {self.performance_stats['total_detections']}")
                print(f"🤖 نمونه‌های آموزشی: {len(self.training_data) if hasattr(self, 'training_data') else 0}")

if __name__ == "__main__":
    # بررسی وجود فایل لازم
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        print("❌ فایل تشخیص چهره پیدا نشد!")
        print("📥 لطفا اول فایل رو دانلود کنید:")
        print("   python download_haar.py")
    else:
        advanced_system = AdvancedEmotionSystem()
        if hasattr(advanced_system, 'cap') and advanced_system.cap is not None and advanced_system.cap.isOpened():
            advanced_system.run_advanced_system()
        else:
            print("❌ سیستم نتوانست راه‌اندازی شود")