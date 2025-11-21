import cv2
import numpy as np
import os
import time

print("🎭 سیستم تشخیص احساسات چهره")
print("=" * 40)

# بررسی وجود فایل تشخیص چهره
if not os.path.exists("haarcascade_frontalface_default.xml"):
    print("❌ فایل تشخیص چهره پیدا نشد!")
    exit()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

print("✅ سیستم آماده است")
print("\n🎯 راهنما:")
print("- صورت خود را جلوی وبکم بگیرید")
print("- برای خروج 'Q' یا 'ESC' بزنید")
print("- یا Ctrl+C در پنجره PowerShell")

start_time = time.time()
should_exit = False

while not should_exit:
    ret, frame = cap.read()
    if not ret:
        break
    
    # تشخیص چهره
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # رسم مستطیل دور چهره‌ها
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # نمایش اطلاعات
    runtime = int(time.time() - start_time)
    cv2.putText(frame, f"Faces: {len(faces)} | Time: {runtime}s", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q or ESC to quit", 
               (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Face Detection", frame)
    
    # مدیریت کلیدها
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
        should_exit = True
        break

# تمیز کردن منابع
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # کمک به بسته شدن پنجره

runtime = int(time.time() - start_time)
print(f"\n✅ برنامه بسته شد! مدت اجرا: {runtime} ثانیه")