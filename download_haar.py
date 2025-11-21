import urllib.request
import os

print(" دانلود فایل تشخیص چهره...")

url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
filename = 'haarcascade_frontalface_default.xml'

try:
    print(" در حال اتصال...")
    urllib.request.urlretrieve(url, filename)
    
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f" فایل دانلود شد! ({file_size} bytes)")
        print(" حالا می‌تونی برنامه رو اجرا کنی:")
        print("   python main.py")
    else:
        print(" خطا در دانلود فایل")
        
except Exception as e:
    print(f" خطا: {e}")
    print("\n راه حل جایگزین:")
    print("1. برو این لینک: https://github.com/opencv/opencv")
    print("2. برو به پوشه: opencv/data/haarcascades/")
    print("3. فایل haarcascade_frontalface_default.xml رو دانلود کن")
    print("4. در پوشه پروژه قرار بده")
