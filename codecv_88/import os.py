import os

file_name = "classification_results.json"
search_path = r"C:\Users\User\OneDrive\Pictures\Screenshots"  # เปลี่ยนเป็นโฟลเดอร์ที่คุณต้องการค้นหา

for root, dirs, files in os.walk(search_path):
    if file_name in files:
        print(f"✅ JSON file found at: {os.path.join(root, file_name)}")
        break
else:
    print("❌ JSON file not found!")
