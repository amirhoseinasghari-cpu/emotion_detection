import requests
import base64
import json
import cv2

class EmotionAPIClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def image_to_base64(self, image_path):
        """تبدیل تصویر به base64"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            print(f"❌ خطا در خواندن تصویر: {e}")
            return None
    
    def analyze_image(self, image_path):
        """ارسال تصویر برای آنالیز"""
        image_data = self.image_to_base64(image_path)
        if not image_data:
            return None
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                json={"image": image_data},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ خطای سرور: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ خطا در ارتباط با سرور: {e}")
            return None
    
    def get_stats(self):
        """دریافت آمار سرور"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except:
            return None
    
    def test_webcam(self):
        """تست با وبکم"""
        cap = cv2.VideoCapture(0)
        print("🎥 وبکم فعال شد. برای خروج 'q' بزنید.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # تبدیل فریم به base64
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{image_data}"
            
            # ارسال برای آنالیز
            try:
                response = requests.post(
                    f"{self.base_url}/analyze",
                    json={"image": image_data},
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result["success"] and result["faces_detected"] > 0:
                        for face in result["analysis"]:
                            x = face["bounding_box"]["x"]
                            y = face["bounding_box"]["y"]
                            w = face["bounding_box"]["width"]
                            h = face["bounding_box"]["height"]
                            
                            # رسم مستطیل
                            color = tuple(face["color"])
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            
                            # نمایش احساسات
                            text = f"{face['emoji']} {face['emotion']} ({face['confidence']:.0%})"
                            cv2.putText(frame, text, (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow('Emotion API Test - Press Q to quit', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except requests.exceptions.RequestException:
                cv2.imshow('Emotion API Test - Press Q to quit', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    client = EmotionAPIClient()
    
    print("🤖 کلاینت API تشخیص احساسات")
    print("1. آنالیز تصویر از فایل")
    print("2. تست با وبکم")
    print("3. نمایش آمار سرور")
    
    choice = input("انتخاب کنید (1/2/3): ").strip()
    
    if choice == "1":
        image_path = input("مسیر تصویر را وارد کنید: ").strip()
        result = client.analyze_image(image_path)
        if result:
            print("✅ نتیجه آنالیز:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("❌ خطا در آنالیز تصویر")
    
    elif choice == "2":
        client.test_webcam()
    
    elif choice == "3":
        stats = client.get_stats()
        if stats:
            print("📊 آمار سرور:")
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            print("❌ خطا در دریافت آمار")
    
    else:
        print("❌ انتخاب نامعتبر")