"""
Script để tải dữ liệu từ Google Drive sử dụng Google Drive API
Sử dụng module data_downloader.py
"""

from data_downloader import GoogleDriveDownloader, DatasetManager

def main():
    """Tải dataset ISIC từ Google Drive"""
    # ID file từ Google Drive link: https://drive.google.com/file/d/1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI/view?usp=sharing
    GOOGLE_DRIVE_FILE_ID = "1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI"

    try:
        print("🚀 Khởi tạo Google Drive Downloader...")

        # Khởi tạo downloader với service account
        downloader = GoogleDriveDownloader(service_account_file='service_account.json')

        # Khởi tạo dataset manager
        dataset_manager = DatasetManager(downloader)

        # Tải và giải nén dataset
        print("\n📥 Bắt đầu tải dataset...")
        success = dataset_manager.download_and_extract_dataset(GOOGLE_DRIVE_FILE_ID)

        if success:
            print("\n🎉 Hoàn thành! Dataset đã được tải và giải nén thành công.")
            print("\n📁 Kiểm tra cấu trúc dữ liệu:")

            import os
            for root, dirs, files in os.walk('data'):
                level = root.replace('data', '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:3]:  # Chỉ hiển thị 3 file đầu tiên
                    print(f"{subindent}{file}")
                if len(files) > 3:
                    print(f"{subindent}... và {len(files) - 3} file khác")
        else:
            print("\n❌ Tải dataset thất bại!")
            print("\n🔧 Hướng dẫn khắc phục:")
            print("1. Kiểm tra file service_account.json có tồn tại và đúng định dạng")
            print("2. Đảm bảo service account có quyền truy cập file trên Google Drive")
            print("3. Kiểm tra kết nối internet")
            print("4. Kiểm tra Google Drive File ID có đúng không")

    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file: {str(e)}")
        print("💡 Đảm bảo file service_account.json tồn tại trong thư mục hiện tại")
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {str(e)}")
        print("💡 Vui lòng kiểm tra lại cấu hình và thử lại")

if __name__ == "__main__":
    main()
