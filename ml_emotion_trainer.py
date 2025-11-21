import cv2
import numpy as np
import os
import time
import json
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MLEmotionTrainer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)
        
        # مدل ماشین لرنینگ
        self.model = None
        self.is_trained = False
        
        # احساسات
        self.emotions = {
            0: {"name": "😠 عصبانی", "color": (0, 0, 255)},
            1: {"name": "😄 شاد", "color": (0, 255, 0)},
            2: {"name": "😢 غمگین", "color": (255, 0, 0)},
            3: {"name": "😲 متعجب", "color": (0, 255, 255)},
            4: {"name": "😐 خنثی", "color": (255, 255, 0)}
        }
        
        # ایجاد پوشه‌ها
        self.setup_folders()
        
        # داده‌های آموزشی
        self.training_data = []
        self.training_labels = []
        
    def setup_folders(self):
        """ایجاد پوشه‌های لازم"""
        folders = ['training_data', 'models', 'datasets']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"✅ پوشه {folder}/ ایجاد شد")
    
    def extract_face_features(self, face_roi):
        """استخراج ویژگی‌های چهره برای مدل ML"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))  # استاندارد کردن سایز
            
            # ویژگی‌های پایه
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # هیستوگرام
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist = hist.flatten()
            
            # ویژگی‌های پیشرفته
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_mean = np.mean(gradient_magnitude)
            
            # ترکیب همه ویژگی‌ها
            features = [
                brightness,
                contrast,
                gradient_mean,
                *hist  # اضافه کردن هیستوگرام
            ]
            
            return np.array(features)
            
        except Exception as e:
            print(f"❌ خطا در استخراج ویژگی‌ها: {e}")
            return None
    
    def collect_training_data(self, face_roi, emotion_id):
        """جمع‌آوری داده برای آموزش مدل"""
        features = self.extract_face_features(face_roi)
        if features is not None:
            self.training_data.append(features)
            self.training_labels.append(emotion_id)
            return True
        return False
    
    def train_model(self):
        """آموزش مدل ماشین لرنینگ"""
        if len(self.training_data) < 10:
            print("❌ داده‌های آموزشی کافی نیست (حداقل ۱۰ نمونه نیاز است)")
            return False
        
        try:
            print("🤖 در حال آموزش مدل...")
            
            # تبدیل به آرایه numpy
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # تقسیم داده‌ها
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # ایجاد و آموزش مدل
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            self.model.fit(X_train, y_train)
            
            # ارزیابی مدل
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            print(f"✅ مدل آموزش داده شد!")
            print(f"📊 دقت مدل: {accuracy:.1%}")
            print(f"📈 تعداد نمونه‌های آموزشی: {len(self.training_data)}")
            
            # ذخیره مدل
            self.save_model()
            return True
            
        except Exception as e:
            print(f"❌ خطا در آموزش مدل: {e}")
            return False
    
    def save_model(self):
        """ذخیره مدل آموزش دیده"""
        if self.model is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"models/emotion_model_{timestamp}.pkl"
        
        try:
            with open(model_filename, 'wb') as f:
                pickle.dump(self.model, f)
            
            # ذخیره metadata
            metadata = {
                "training_date": datetime.now().isoformat(),
                "training_samples": len(self.training_data),
                "emotions": self.emotions,
                "feature_dimension": self.training_data[0].shape[0] if self.training_data else 0
            }
            
            with open(f"models/model_metadata_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"💾 مدل ذخیره شد: {model_filename}")
            
        except Exception as e:
            print(f"❌ خطا در ذخیره مدل: {e}")
    
    def load_model(self, model_path):
        """بارگذاری مدل از فایل"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_trained = True
            print(f"✅ مدل بارگذاری شد: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ خطا در بارگذاری مدل: {e}")
            return False
    
    def predict_emotion_ml(self, face_roi):
        """پیش‌بینی احساسات با مدل ML"""
        if not self.is_trained or self.model is None:
            return self.predict_emotion_basic(face_roi)
        
        try:
            features = self.extract_face_features(face_roi)
            if features is None:
                return 4, 0.5  # حالت پیش‌فرض
            
            # پیش‌بینی
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = probabilities[prediction]
            
            return prediction, confidence
            
        except Exception as e:
            print(f"❌ خطا در پیش‌بینی: {e}")
            return self.predict_emotion_basic(face_roi)
    
    def predict_emotion_basic(self, face_roi):
        """پیش‌بینی احساسات با روش پایه (Fallback)"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness > 170:
                return 1, 0.8  # شاد
            elif brightness < 100:
                return 2, 0.7  # غمگین
            else:
                return 4, 0.6  # خنثی
                
        except:
            return 4, 0.5
    
    def run(self):
        """اجرای برنامه اصلی"""
        if self.face_cascade.empty() or not self.cap.isOpened():
            print("❌ خطا در راه‌اندازی سیستم")
            return
        
        print("🤖 سیستم تشخیص احساسات با ماشین لرنینگ")
        print("=" * 55)
        print("✅ سیستم آماده است")
        print("\n🎯 راهنما:")
        print("- 0-4: برچسب‌گذاری احساسات (0=عصبانی, 1=شاد, 2=غمگین, 3=متعجب, 4=خنثی)")
        print("- T: آموزش مدل با داده‌های جمع‌آوری شده")
        print("- P: پیش‌بینی با مدل آموزش دیده")
        print("- S: ذخیره داده‌های آموزشی")
        print("- Q: خروج")
        
        current_emotion_label = 4  # پیش‌فرض: خنثی
        use_ml_prediction = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # تشخیص چهره
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # پیش‌بینی احساسات
                if use_ml_prediction and self.is_trained:
                    emotion_id, confidence = self.predict_emotion_ml(face_roi)
                    prediction_source = "ML"
                else:
                    emotion_id, confidence = self.predict_emotion_basic(face_roi)
                    prediction_source = "Basic"
                
                emotion_data = self.emotions[emotion_id]
                
                # رسم مستطیل و نمایش اطلاعات
                cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_data["color"], 3)
                
                emotion_text = f"{emotion_data['name']} ({confidence:.0%})"
                cv2.putText(frame, emotion_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_data["color"], 2)
                
                # منبع پیش‌بینی
                source_text = f"Source: {prediction_source}"
                cv2.putText(frame, source_text, (x, y+h+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_data["color"], 1)
            
            # نمایش اطلاعات
            stats_text = f"چهره‌ها: {len(faces)} | داده‌های آموزشی: {len(self.training_data)}"
            cv2.putText(frame, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # وضعیت مدل
            model_status = "مدل: آموزش دیده" if self.is_trained else "مدل: آموزش ندیده"
            cv2.putText(frame, model_status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.is_trained else (0, 0, 255), 2)
            
            # حالت پیش‌بینی
            pred_mode = "پیش‌بینی: ML" if use_ml_prediction else "پیش‌بینی: Basic"
            cv2.putText(frame, pred_mode, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # راهنما
            help_text = "0-4:برچسب T:آموزش P:ML/S:ذخیره Q:خروج"
            cv2.putText(frame, help_text, (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("🤖 تشخیص احساسات با ML", frame)
            
            # مدیریت کلیدها
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break
            elif ord('0') <= key <= ord('4'):
                emotion_id = key - ord('0')
                if faces:
                    face_roi = frame[y:y+h, x:x+w]
                    if self.collect_training_data(face_roi, emotion_id):
                        emotion_name = self.emotions[emotion_id]["name"]
                        print(f"🏷️  برچسب‌گذاری شد: {emotion_name}")
            elif key == ord('t') or key == ord('T'):
                self.train_model()
            elif key == ord('p') or key == ord('P'):
                use_ml_prediction = not use_ml_prediction
                status = "فعال" if use_ml_prediction else "غیرفعال"
                print(f"🔧 پیش‌بینی ML {status} شد")
            elif key == ord('s') or key == ord('S'):
                self.save_training_dataset()
        
        # تمیز کردن
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ برنامه پایان یافت!")
        print(f"📊 داده‌های آموزشی جمع‌آوری شده: {len(self.training_data)}")
    
    def save_training_dataset(self):
        """ذخیره داده‌های آموزشی"""
        if not self.training_data:
            print("❌ هیچ داده‌ای برای ذخیره وجود ندارد")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_file = f"datasets/training_data_{timestamp}.npz"
        
        try:
            np.savez_compressed(
                dataset_file,
                data=self.training_data,
                labels=self.training_labels
            )
            print(f"💾 داده‌های آموزشی ذخیره شد: {dataset_file}")
            print(f"📈 تعداد نمونه‌ها: {len(self.training_data)}")
            
        except Exception as e:
            print(f"❌ خطا در ذخیره داده‌ها: {e}")

if __name__ == "__main__":
    trainer = MLEmotionTrainer()
    trainer.run()