import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import time
import base64
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import warnings
warnings.filterwarnings('ignore')

class EmotionReportSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)
        
        # داده‌های تاریخی
        self.session_data = []
        self.daily_stats = {}
        
        # احساسات
        self.emotions = {
            0: {"name": "عصبانی", "emoji": "😠", "color": "#FF4444"},
            1: {"name": "شاد", "emoji": "😄", "color": "#44FF44"}, 
            2: {"name": "غمگین", "emoji": "😢", "color": "#4444FF"},
            3: {"name": "متعجب", "emoji": "😲", "color": "#FFFF44"},
            4: {"name": "خنثی", "emoji": "😐", "color": "#888888"}
        }
        
        # ایجاد پوشه‌ها
        self.setup_folders()
        
        # آمار جلسه
        self.session_start = datetime.now()
        self.current_emotions = []
        
    def setup_folders(self):
        """ایجاد پوشه‌های مورد نیاز"""
        folders = [
            'reports/pdf',
            'reports/excel', 
            'reports/charts',
            'reports/session_data',
            'exports'
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
    
    def collect_data(self, emotion_id, confidence, face_size):
        """جمع‌آوری داده‌های تشخیص"""
        data_point = {
            'timestamp': datetime.now(),
            'emotion_id': emotion_id,
            'emotion_name': self.emotions[emotion_id]['name'],
            'confidence': confidence,
            'face_size': face_size,
            'session_duration': (datetime.now() - self.session_start).total_seconds()
        }
        
        self.session_data.append(data_point)
        self.current_emotions.append(emotion_id)
        
        # محدود کردن تاریخچه به 1000 رکورد
        if len(self.current_emotions) > 1000:
            self.current_emotions.pop(0)
    
    def generate_excel_report(self):
        """ایجاد گزارش اکسل"""
        try:
            if not self.session_data:
                return None
                
            # ایجاد DataFrame
            df = pd.DataFrame(self.session_data)
            
            # محاسبه آمار
            summary_stats = {
                'تاریخ تولید': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'مدت زمان جلسه (دقیقه)': round((datetime.now() - self.session_start).total_seconds() / 60, 2),
                'تعداد تشخیص‌ها': len(self.session_data),
                'میانگین اطمینان': round(df['confidence'].mean() * 100, 2),
                'احساس غالب': df['emotion_name'].mode()[0] if not df.empty else 'N/A'
            }
            
            # آمار احساسات
            emotion_stats = df['emotion_name'].value_counts().to_dict()
            
            # ایجاد فایل اکسل
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/excel/emotion_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # داده‌های خام
                df.to_excel(writer, sheet_name='داده‌های خام', index=False)
                
                # آمار کلی
                summary_df = pd.DataFrame([summary_stats])
                summary_df.to_excel(writer, sheet_name='آمار کلی', index=False)
                
                # آمار احساسات
                emotion_df = pd.DataFrame(list(emotion_stats.items()), columns=['احساس', 'تعداد'])
                emotion_df.to_excel(writer, sheet_name='توزیع احساسات', index=False)
                
                # روند زمانی
                df['time_minutes'] = df['session_duration'] / 60
                time_stats = df.groupby(pd.cut(df['time_minutes'], bins=10))['emotion_name'].agg(lambda x: x.mode()[0] if not x.empty else 'N/A')
                time_stats.to_excel(writer, sheet_name='روند زمانی')
            
            print(f"📊 گزارش اکسل ایجاد شد: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ خطا در ایجاد گزارش اکسل: {e}")
            return None
    
    def generate_charts(self):
        """ایجاد نمودارهای مختلف"""
        if not self.session_data:
            return None
            
        df = pd.DataFrame(self.session_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_files = []
        
        try:
            # تنظیم استایل
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # نمودار 1: توزیع احساسات
            plt.figure(figsize=(10, 6))
            emotion_counts = df['emotion_name'].value_counts()
            colors = [self.emotions[i]['color'] for i in range(5) if self.emotions[i]['name'] in emotion_counts.index]
            
            plt.subplot(2, 2, 1)
            bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors, alpha=0.7)
            plt.title('توزیع احساسات')
            plt.xticks(rotation=45)
            
            # اضافه کردن اعداد روی نمودار
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # نمودار 2: روند اطمینان
            plt.subplot(2, 2, 2)
            df['time_minutes'] = df['session_duration'] / 60
            plt.scatter(df['time_minutes'], df['confidence'] * 100, alpha=0.6)
            plt.xlabel('زمان (دقیقه)')
            plt.ylabel('اطمینان (%)')
            plt.title('روند اطمینان تشخیص')
            
            # نمودار 3: میانگین اطمینان بر اساس احساس
            plt.subplot(2, 2, 3)
            confidence_by_emotion = df.groupby('emotion_name')['confidence'].mean() * 100
            plt.bar(confidence_by_emotion.index, confidence_by_emotion.values, alpha=0.7)
            plt.title('میانگین اطمینان بر اساس احساس')
            plt.xticks(rotation=45)
            
            # نمودار 4: توزیع اندازه چهره
            plt.subplot(2, 2, 4)
            plt.hist(df['face_size'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('اندازه چهره')
            plt.ylabel('تعداد')
            plt.title('توزیع اندازه چهره‌ها')
            
            plt.tight_layout()
            chart_path = f"reports/charts/charts_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            chart_files.append(chart_path)
            
            # نمودار 5: نمودار حرارتی زمانی
            plt.figure(figsize=(12, 6))
            df['time_bin'] = pd.cut(df['time_minutes'], bins=20)
            heatmap_data = pd.crosstab(df['time_bin'], df['emotion_name'])
            sns.heatmap(heatmap_data.T, cmap='YlOrRd', annot=True, fmt='d')
            plt.title('توزیع احساسات در طول زمان')
            plt.xlabel('بازه زمانی (دقیقه)')
            plt.ylabel('احساسات')
            
            heatmap_path = f"reports/charts/heatmap_{timestamp}.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            chart_files.append(heatmap_path)
            
            print(f"📈 نمودارها ایجاد شدند")
            return chart_files
            
        except Exception as e:
            print(f"❌ خطا در ایجاد نمودارها: {e}")
            return None
    
    def generate_pdf_report(self):
        """ایجاد گزارش PDF حرفه‌ای"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"reports/pdf/emotion_report_{timestamp}.pdf"
            
            # ایجاد PDF
            c = canvas.Canvas(pdf_filename, pagesize=A4)
            width, height = A4
            
            # هدر
            c.setFont("Helvetica-Bold", 18)
            c.drawString(100, height - 100, "گزارش تحلیل احساسات چهره")
            c.setFont("Helvetica", 12)
            c.drawString(100, height - 130, f"تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # آمار کلی
            c.setFont("Helvetica-Bold", 14)
            c.drawString(100, height - 180, "آمار کلی جلسه:")
            c.setFont("Helvetica", 10)
            
            if self.session_data:
                df = pd.DataFrame(self.session_data)
                stats = [
                    f"مدت زمان جلسه: {round((datetime.now() - self.session_start).total_seconds() / 60, 1)} دقیقه",
                    f"تعداد تشخیص‌ها: {len(self.session_data)}",
                    f"میانگین اطمینان: {round(df['confidence'].mean() * 100, 1)}%",
                    f"احساس غالب: {df['emotion_name'].mode()[0] if not df.empty else 'N/A'}"
                ]
                
                y_pos = height - 210
                for stat in stats:
                    c.drawString(120, y_pos, stat)
                    y_pos -= 20
            
            # اضافه کردن نمودارها
            chart_files = self.generate_charts()
            if chart_files:
                y_pos = height - 300
                for chart_file in chart_files:
                    try:
                        img = ImageReader(chart_file)
                        c.drawImage(img, 50, y_pos - 200, width=500, height=200)
                        y_pos -= 250
                    except:
                        continue
            
            # جدول توزیع احساسات
            if self.session_data:
                c.showPage()  # صفحه جدید
                c.setFont("Helvetica-Bold", 14)
                c.drawString(100, height - 100, "توزیع دقیق احساسات:")
                
                df = pd.DataFrame(self.session_data)
                emotion_dist = df['emotion_name'].value_counts()
                
                y_pos = height - 140
                c.setFont("Helvetica", 10)
                for emotion, count in emotion_dist.items():
                    percentage = (count / len(self.session_data)) * 100
                    c.drawString(120, y_pos, f"{emotion}: {count} نمونه ({percentage:.1f}%)")
                    y_pos -= 20
            
            c.save()
            print(f"📄 گزارش PDF ایجاد شد: {pdf_filename}")
            return pdf_filename
            
        except Exception as e:
            print(f"❌ خطا در ایجاد گزارش PDF: {e}")
            return None
    
    def generate_dashboard_data(self):
        """ایجاد داده‌های دشبورد"""
        if not self.session_data:
            return None
            
        df = pd.DataFrame(self.session_data)
        
        dashboard_data = {
            'summary': {
                'total_detections': len(self.session_data),
                'session_duration_minutes': round((datetime.now() - self.session_start).total_seconds() / 60, 1),
                'average_confidence': round(df['confidence'].mean() * 100, 1),
                'dominant_emotion': df['emotion_name'].mode()[0] if not df.empty else 'N/A'
            },
            'emotion_distribution': df['emotion_name'].value_counts().to_dict(),
            'confidence_trend': {
                'time': [d['session_duration'] / 60 for d in self.session_data],
                'confidence': [d['confidence'] * 100 for d in self.session_data]
            },
            'realtime_emotions': self.current_emotions[-20:]  # آخرین 20 تشخیص
        }
        
        # ذخیره داده‌های دشبورد
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"exports/dashboard_data_{timestamp}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
        
        print(f"📋 داده‌های دشبورد ذخیره شد: {json_filename}")
        return dashboard_data
    
    def send_email_report(self, recipient_email, subject="گزارش تحلیل احساسات"):
        """ارسال گزارش از طریق ایمیل"""
        try:
            # ایجاد گزارش‌ها
            pdf_file = self.generate_pdf_report()
            excel_file = self.generate_excel_report()
            
            if not pdf_file or not excel_file:
                print("❌ خطا در ایجاد فایل‌های گزارش")
                return False
            
            # تنظیمات ایمیل (این بخش نیاز به تنظیم دارد)
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "your_email@gmail.com"  # باید تنظیم شود
            sender_password = "your_app_password"  # باید تنظیم شود
            
            # ایجاد ایمیل
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # متن ایمیل
            body = f"""
            گزارش تحلیل احساسات چهره
            
            تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            تعداد تشخیص‌ها: {len(self.session_data)}
            
            فایل‌های پیوست شده شامل:
            - گزارش PDF کامل
            - داده‌های خام در قالب Excel
            
            با تشکر
            سیستم تحلیل احساسات
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # پیوست کردن فایل‌ها
            for file_path in [pdf_file, excel_file]:
                with open(file_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                filename = os.path.basename(file_path)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {filename}",
                )
                msg.attach(part)
            
            # ارسال ایمیل
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"📧 گزارش به {recipient_email} ارسال شد")
            return True
            
        except Exception as e:
            print(f"❌ خطا در ارسال ایمیل: {e}")
            return False
    
    def real_time_analysis(self):
        """آنالیز لحظه‌ای و جمع‌آوری داده"""
        print("🎯 سیستم گزارش‌گیری فعال - در حال جمع‌آوری داده...")
        print("دستورات:")
        print("  R: ایجاد گزارش لحظه‌ای")
        print("  E: خروج و ایجاد گزارش نهایی")
        print("  D: نمایش دشبورد داده‌ها")
        
        start_time = time.time()
        analysis_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # تشخیص چهره
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotion_id, confidence = self.analyze_emotion(face_roi)
                face_size = w * h
                
                # جمع‌آوری داده
                self.collect_data(emotion_id, confidence, face_size)
                analysis_count += 1
            
            # نمایش اطلاعات
            cv2.putText(frame, f"تحلیل‌ها: {analysis_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "R:گزارش E:خروج D:دشبورد", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Real-time Analysis - Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or key == ord('R'):
                print("📊 در حال ایجاد گزارش لحظه‌ای...")
                self.generate_pdf_report()
                self.generate_excel_report()
            elif key == ord('d') or key == ord('D'):
                dashboard = self.generate_dashboard_data()
                if dashboard:
                    print("📋 داده‌های دشبورد آماده است")
            elif key == ord('e') or key == ord('E'):
                break
        
        # گزارش نهایی
        print("📦 در حال ایجاد گزارش‌های نهایی...")
        self.generate_pdf_report()
        self.generate_excel_report()
        self.generate_dashboard_data()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        duration = time.time() - start_time
        print(f"\n✅ جمع‌آوری داده پایان یافت!")
        print(f"📊 آمار نهایی:")
        print(f"   مدت زمان: {duration:.1f} ثانیه")
        print(f"   تعداد تحلیل‌ها: {analysis_count}")
        print(f"   گزارش‌ها در پوشه reports/ ذخیره شدند")
    
    def analyze_emotion(self, face_roi):
        """آنالیز احساسات"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness > 170:
                return 1, 0.85  # شاد
            elif brightness < 100:
                return 2, 0.75  # غمگین
            elif brightness > 200:
                return 3, 0.80  # متعجب
            else:
                return 4, 0.70  # خنثی
        except:
            return 4, 0.5

if __name__ == "__main__":
    report_system = EmotionReportSystem()
    
    print("📊 سیستم پیشرفته گزارش‌گیری احساسات")
    print("=" * 50)
    print("1. جمع‌آوری داده و گزارش لحظه‌ای")
    print("2. ایجاد گزارش از داده‌های موجود")
    print("3. دشبورد داده‌ها")
    
    choice = input("انتخاب کنید (1/2/3): ").strip()
    
    if choice == "1":
        report_system.real_time_analysis()
    elif choice == "2":
        if report_system.session_data:
            report_system.generate_pdf_report()
            report_system.generate_excel_report()
            print("✅ گزارش‌ها ایجاد شدند")
        else:
            print("❌ داده‌ای برای گزارش‌گیری وجود ندارد")
    elif choice == "3":
        dashboard = report_system.generate_dashboard_data()
        if dashboard:
            print("📋 دشبورد داده‌ها آماده است")
    else:
        print("❌ انتخاب نامعتبر")