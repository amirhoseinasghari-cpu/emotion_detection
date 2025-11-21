import os
import shutil

def cleanup_project():
    print("🧹 در حال تمیز کردن پوشه‌های پروژه...")
    print("=" * 50)
    
    # پوشه‌هایی که باید نگه داشته بشن
    essential_folders = [
        'assets',           # برای عکس‌ها و فایل‌های استاتیک
        'models',           # مدل‌های AI
        'src'               # کدهای اصلی
    ]
    
    # پوشه‌هایی که باید حذف بشن (اگر خالی هستند)
    folders_to_remove_if_empty = [
        'advanced_data',
        'advanced_exports', 
        'advanced_reports',
        'api_logs',
        'charts',
        'datasets',
        'data_logs',
        'exports',
        'reports',
        'saved_faces',
        'screenshots',
        'sessions', 
        'training_data',
        'web_charts',
        'web_exports',
        'web_results'
    ]
    
    # فایل‌های ضروری که باید نگه داشته بشن
    essential_files = [
        'haarcascade_frontalface_default.xml',
        'requirements.txt',
        'main.py',
        'web_emotion_app.py',
        'download_haar.py'
    ]
    
    # حذف پوشه‌های خالی
    print("\n🗑️ حذف پوشه‌های خالی:")
    removed_folders = 0
    for folder in folders_to_remove_if_empty:
        if os.path.exists(folder):
            try:
                # فقط اگر پوشه خالی هست حذف کن
                if not os.listdir(folder):
                    os.rmdir(folder)
                    print(f"  ✅ {folder}/ (خالی) حذف شد")
                    removed_folders += 1
                else:
                    print(f"  📁 {folder}/ (غیرخالی) نگه داشته شد")
            except Exception as e:
                print(f"  ❌ {folder}/: {e}")
    
    # نمایش پوشه‌های باقی‌مانده
    print(f"\n📊 نتیجه:")
    print(f"  پوشه‌های حذف شده: {removed_folders}")
    
    # نمایش ساختار نهایی
    print(f"\n🏗️ ساختار نهایی پروژه:")
    current_files = os.listdir('.')
    folders = [f for f in current_files if os.path.isdir(f)]
    files = [f for f in current_files if os.path.isfile(f) and f.endswith('.py')]
    
    print("پوشه‌ها:")
    for folder in sorted(folders):
        size = len(os.listdir(folder)) if os.path.exists(folder) else 0
        print(f"  📁 {folder}/ ({size} فایل)")
    
    print("\nفایل‌های پایتون:")
    for file in sorted(files):
        size = os.path.getsize(file)
        print(f"  📄 {file} ({size} بایت)")
    
    print(f"\n✅ تمیزکاری کامل شد!")
    print(f"💡 برای GitHub Pages فقط فایل index.html نیاز داری")

def create_minimal_structure():
    """ایجاد ساختار مینیمال برای GitHub Pages"""
    print("\n🎯 ایجاد ساختار برای GitHub Pages...")
    
    # فایل‌های مورد نیاز برای GitHub Pages
    github_files = ['index.html', 'README.md', 'assets/']
    
    print("فایل‌های ضروری برای آپلود:")
    for file in github_files:
        print(f"  📄 {file}")
    
    print("\n🎯 دستور بعدی:")
    print("1. فایل index.html رو ایجاد کن")
    print("2. روی GitHub آپلود کن")
    print("3. GitHub Pages رو فعال کن")

if __name__ == "__main__":
    cleanup_project()
    create_minimal_structure()