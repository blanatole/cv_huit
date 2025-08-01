"""
Module để tải dữ liệu từ Google Drive sử dụng Google Drive API
"""

import os
import io
import zipfile
import rarfile
import subprocess
import platform
from typing import Optional, Tuple
from tqdm import tqdm

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError


class GoogleDriveDownloader:
    """Class để tải file từ Google Drive sử dụng Service Account"""
    
    def __init__(self, service_account_file: str = 'service_account.json'):
        """
        Khởi tạo Google Drive downloader
        
        Args:
            service_account_file: Đường dẫn tới file JSON service account
        """
        self.service_account_file = service_account_file
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Xác thực với Google Drive API"""
        try:
            if not os.path.exists(self.service_account_file):
                raise FileNotFoundError(f"Service account file không tồn tại: {self.service_account_file}")
            
            # Xác thực
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_file, scopes=self.scopes)
            
            # Tạo service Drive
            self.service = build('drive', 'v3', credentials=creds)
            print("✅ Xác thực Google Drive API thành công")
            
        except Exception as e:
            print(f"❌ Lỗi xác thực Google Drive API: {str(e)}")
            raise
    
    def get_file_info(self, file_id: str) -> Optional[dict]:
        """
        Lấy thông tin file từ Google Drive
        
        Args:
            file_id: ID của file trên Google Drive
            
        Returns:
            Dict chứa thông tin file hoặc None nếu lỗi
        """
        try:
            file_info = self.service.files().get(fileId=file_id).execute()
            return file_info
        except HttpError as e:
            print(f"❌ Lỗi khi lấy thông tin file: {str(e)}")
            return None
    
    def download_file(self, file_id: str, output_path: str, show_progress: bool = True) -> bool:
        """
        Tải file từ Google Drive

        Args:
            file_id: ID của file trên Google Drive
            output_path: Đường dẫn lưu file
            show_progress: Hiển thị thanh tiến trình

        Returns:
            True nếu tải thành công, False nếu thất bại
        """
        try:
            # Lấy thông tin file
            file_info = self.get_file_info(file_id)
            if not file_info:
                return False

            file_name = file_info.get('name', 'unknown_file')
            file_size = file_info.get('size')

            print(f"📥 Đang tải file: {file_name}")

            # Kiểm tra kích thước file
            if file_size and file_size.isdigit():
                file_size_int = int(file_size)
                print(f"📊 Kích thước: {self._format_size(file_size_int)}")
            else:
                file_size_int = 0
                print(f"📊 Kích thước: Không xác định (file có thể được nén hoặc là Google Docs)")

            # Tạo thư mục nếu cần
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Gửi yêu cầu tải file
            request = self.service.files().get_media(fileId=file_id)

            # Tạo file để ghi
            with open(output_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)

                if show_progress and file_size_int > 0:
                    # Sử dụng tqdm với kích thước file biết trước
                    pbar = tqdm(total=file_size_int, unit='B', unit_scale=True, desc="Downloading")

                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        if status:
                            downloaded = int(status.resumable_progress)
                            pbar.update(downloaded - pbar.n)

                    pbar.close()
                elif show_progress:
                    # Sử dụng tqdm không biết kích thước file
                    print("📥 Đang tải... (hiển thị tiến trình theo %)")

                    done = False
                    last_progress = 0
                    while not done:
                        try:
                            status, done = downloader.next_chunk()
                            if status:
                                current_progress = int(status.progress() * 100)
                                if current_progress > last_progress:
                                    print(f"📥 Tiến trình: {current_progress}%")
                                    last_progress = current_progress
                        except Exception as chunk_error:
                            print(f"⚠️ Lỗi khi tải chunk: {str(chunk_error)}")
                            # Thử tiếp tục
                            continue
                else:
                    # Tải không có thanh tiến trình
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        if status and show_progress:
                            print(f"Đang tải: {int(status.progress() * 100)}%")

            # Kiểm tra kích thước file đã tải
            if os.path.exists(output_path):
                actual_size = os.path.getsize(output_path)
                print(f"✅ Tải file thành công: {output_path}")
                print(f"📊 Kích thước thực tế: {self._format_size(actual_size)}")
            else:
                print(f"❌ File không được tạo: {output_path}")
                return False

            return True

        except Exception as e:
            print(f"❌ Lỗi khi tải file: {str(e)}")
            return False
    
    def _format_size(self, size_bytes: int) -> str:
        """Định dạng kích thước file"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"


class DatasetManager:
    """Class để quản lý việc tải và giải nén dataset"""
    
    def __init__(self, downloader: GoogleDriveDownloader):
        self.downloader = downloader
    
    def setup_data_structure(self):
        """Tạo cấu trúc thư mục data"""
        directories = [
            'data',
            'data/train',
            'data/train/images',
            'data/train/ground_truth',
            'data/val',
            'data/val/images', 
            'data/val/ground_truth',
            'data/test',
            'data/test/images',
            'data/test/ground_truth'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Tạo thư mục: {directory}")
    
    def install_unrar_if_needed(self) -> bool:
        """Cài đặt unrar nếu cần thiết trên Linux"""
        system = platform.system().lower()
        
        if system == 'windows':
            return True  # Windows không cần unrar command line
        
        # Kiểm tra xem unrar đã được cài đặt chưa
        try:
            subprocess.run(['unrar'], capture_output=True, check=False)
            return True
        except FileNotFoundError:
            pass
        
        # Thử cài đặt unrar
        print("🔧 Đang cài đặt unrar...")
        try:
            # Thử cài đặt với apt (Ubuntu/Debian)
            subprocess.run(['sudo', 'apt', 'update'], check=True, capture_output=True)
            subprocess.run(['sudo', 'apt', 'install', '-y', 'unrar'], check=True)
            print("✅ Đã cài đặt unrar thành công")
        except subprocess.CalledProcessError:
            try:
                # Thử cài đặt với yum (CentOS/RHEL)
                subprocess.run(['sudo', 'yum', 'install', '-y', 'unrar'], check=True)
                print("✅ Đã cài đặt unrar thành công")
            except subprocess.CalledProcessError:
                print("❌ Không thể cài đặt unrar tự động. Vui lòng cài đặt thủ công:")
                print("   Ubuntu/Debian: sudo apt install unrar")
                print("   CentOS/RHEL: sudo yum install unrar")
                return False
        return True
    
    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        """
        Giải nén file zip hoặc rar
        
        Args:
            archive_path: Đường dẫn file nén
            extract_to: Thư mục giải nén
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        print(f"📂 Đang giải nén {archive_path}...")
        
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.endswith('.rar'):
                try:
                    with rarfile.RarFile(archive_path, 'r') as rar_ref:
                        rar_ref.extractall(extract_to)
                except rarfile.Error:
                    # Fallback to command line unrar
                    try:
                        subprocess.run(['unrar', 'x', archive_path, extract_to], check=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        print("❌ Không thể giải nén file RAR. Cần cài đặt unrar hoặc rarfile")
                        print("   Cài đặt: pip install rarfile")
                        print("   Hoặc: sudo apt install unrar (Linux)")
                        return False
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {archive_path}")
            
            print(f"✅ Đã giải nén thành công vào {extract_to}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi giải nén: {str(e)}")
            return False
    
    def download_and_extract_dataset(self, file_id: str, temp_dir: str = 'temp_downloads') -> bool:
        """
        Tải và giải nén dataset từ Google Drive
        
        Args:
            file_id: ID file trên Google Drive
            temp_dir: Thư mục tạm để lưu file tải về
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        print("🚀 Bắt đầu tải ISIC dataset từ Google Drive...")
        print("⚠️  Lưu ý: Quá trình này có thể mất vài phút tùy thuộc vào tốc độ mạng")
        
        # Kiểm tra xem dữ liệu đã tồn tại chưa
        if os.path.exists('data/train/images') and len(os.listdir('data/train/images')) > 0:
            print("✅ Dữ liệu đã tồn tại, bỏ qua việc tải")
            return True
        
        try:
            # Cài đặt unrar nếu cần thiết (Linux)
            if not self.install_unrar_if_needed():
                return False
            
            # Tạo thư mục tạm
            os.makedirs(temp_dir, exist_ok=True)
            
            # Lấy thông tin file để xác định extension
            file_info = self.downloader.get_file_info(file_id)
            if not file_info:
                print("❌ Không thể lấy thông tin file")
                return False
            
            file_name = file_info.get('name', 'data.rar')
            output_path = os.path.join(temp_dir, file_name)
            
            # Tải file từ Google Drive
            print(f"\n📥 Đang tải {file_name} từ Google Drive...")
            if not self.downloader.download_file(file_id, output_path):
                return False
            
            # Tạo cấu trúc thư mục
            print("\n📁 Tạo cấu trúc thư mục...")
            self.setup_data_structure()
            
            # Giải nén file
            print(f"\n📂 Đang giải nén {file_name}...")
            if not self.extract_archive(output_path, '.'):
                return False
            
            # Xóa file nén sau khi giải nén
            os.remove(output_path)
            
            # Xóa thư mục tạm
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
            print("\n🎉 Dataset download và extract hoàn thành!")
            print("📁 Cấu trúc dữ liệu:")
            print("   data/")
            print("   ├── train/images/")
            print("   ├── train/ground_truth/")
            print("   ├── val/images/")
            print("   ├── val/ground_truth/")
            print("   ├── test/images/")
            print("   └── test/ground_truth/")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
            print("\n🔧 Hướng dẫn khắc phục:")
            print("1. Kiểm tra Google Drive File ID có đúng không")
            print("2. Đảm bảo file được chia sẻ với service account")
            print("3. Kiểm tra file service_account.json có đúng không")
            print("4. Kiểm tra kết nối internet")
            print("5. Thử tải thủ công và giải nén vào thư mục data/")
            return False


def main():
    """Hàm main để test"""
    # ID file từ Google Drive link
    GOOGLE_DRIVE_FILE_ID = "1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI"
    
    try:
        # Khởi tạo downloader
        downloader = GoogleDriveDownloader()
        
        # Khởi tạo dataset manager
        dataset_manager = DatasetManager(downloader)
        
        # Tải và giải nén dataset
        if dataset_manager.download_and_extract_dataset(GOOGLE_DRIVE_FILE_ID):
            print("\n📝 Bước tiếp theo:")
            print("1. Kiểm tra dữ liệu đã được tải đúng")
            print("2. Bắt đầu training models!")
        else:
            print("\n❌ Tải dữ liệu thất bại. Vui lòng kiểm tra lại hoặc tải thủ công.")
            
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")


if __name__ == "__main__":
    main()
