"""
Test script để kiểm tra cell tải dữ liệu trong notebook
"""

import os
from data_downloader import GoogleDriveDownloader, DatasetManager

# ID file từ Google Drive link: https://drive.google.com/file/d/1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI/view?usp=sharing
GOOGLE_DRIVE_FILE_ID = "1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI"

def download_and_extract_data():
    """Tải và giải nén dữ liệu từ Google Drive sử dụng Google Drive API"""
    try:
        # Khởi tạo Google Drive downloader
        print("🔑 Khởi tạo Google Drive API...")
        downloader = GoogleDriveDownloader(service_account_file='service_account.json')
        
        # Khởi tạo dataset manager
        dataset_manager = DatasetManager(downloader)
        
        # Tải và giải nén dataset
        success = dataset_manager.download_and_extract_dataset(GOOGLE_DRIVE_FILE_ID)
        
        return success
        
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file: {str(e)}")
        print("💡 Đảm bảo file service_account.json tồn tại trong thư mục hiện tại")
        print("\n🔧 Hướng dẫn khắc phục:")
        print("1. Tạo service account trên Google Cloud Console")
        print("2. Tải file JSON credentials và đặt tên là 'service_account.json'")
        print("3. Chia sẻ Google Drive file với email service account")
        return False
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {str(e)}")
        print("💡 Vui lòng kiểm tra lại cấu hình và thử lại")
        return False

# Thực hiện tải và giải nén dữ liệu
if __name__ == "__main__":
    print("🧪 Test notebook cell - Tải dữ liệu từ Google Drive API")
    print("=" * 60)
    
    if download_and_extract_data():
        print("\n📝 Bước tiếp theo:")
        print("1. Kiểm tra dữ liệu đã được tải đúng")
        print("2. Bắt đầu training models!")
    else:
        print("\n❌ Tải dữ liệu thất bại. Vui lòng kiểm tra lại hoặc tải thủ công.")
