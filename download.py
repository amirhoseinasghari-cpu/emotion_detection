import urllib.request
import os

print(" دانلود فایل تشخیص چهره...")
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
filename = "haarcascade_frontalface_default.xml"

try:
    urllib.request.urlretrieve(url, filename)
    print(" فایل دانلود شد!")
except Exception as e:
    print(f" خطا: {e}")
