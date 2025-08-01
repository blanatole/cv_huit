# 🚀 Setup Guide - Thiết lập Project trên Thiết bị Mới

Hướng dẫn này giúp bạn thiết lập project phân đoạn khối u trên thiết bị mới sau khi clone từ GitHub.

## ✨ **Cập nhật mới**: Tải dữ liệu tự động trong Notebook!
**Không cần setup phức tạp** - Notebook sẽ tự động tải và chuẩn bị dữ liệu cho bạn!

## � **TL;DR - Setup siêu nhanh**
```bash
git clone https://github.com/blanatole/CV_Master.git
cd CV_Master
jupyter notebook medical_tumor_segmentation.ipynb
# Chạy các cell từ trên xuống dưới - Done! 🎉
```

## �📋 Yêu cầu hệ thống

### Hardware:
- **GPU**: NVIDIA GPU với ít nhất 8GB VRAM (khuyến nghị)
- **RAM**: Ít nhất 16GB
- **Storage**: Ít nhất 15GB trống (10GB cho data + 5GB cho models)

### Software:
- Python 3.8+
- CUDA 11.0+ (nếu sử dụng GPU)
- Git

## 🎯 **Lợi ích phiên bản mới**
- ✅ **Zero-config setup**: Không cần cài đặt gì trước
- ✅ **Tự động tải dữ liệu**: Từ Google Drive với progress tracking
- ✅ **Hỗ trợ đa nền tảng**: Windows, Linux, macOS
- ✅ **Tự động cài unrar**: Trên Linux nếu cần
- ✅ **Kiểm tra dữ liệu**: Tự động verify tính toàn vẹn
- ✅ **Một workflow duy nhất**: Chỉ cần chạy notebook

## 🔧 Các bước thiết lập

### 1. Clone Repository
```bash
git clone https://github.com/blanatole/CV_Master.git
cd CV_Master
```

### 2. Tạo Virtual Environment (Khuyến nghị)
```bash
# Sử dụng conda
conda create -n cv_master python=3.9
conda activate cv_master

# Hoặc sử dụng venv
python -m venv cv_master_env
# Windows:
cv_master_env\Scripts\activate
# Linux/Mac:
source cv_master_env/bin/activate
```

### 3. Chạy Notebook (Tự động cài đặt và tải dữ liệu)
```bash
jupyter notebook medical_tumor_segmentation.ipynb
```

**🎉 Đó là tất cả!** Notebook sẽ tự động:
- Cài đặt tất cả dependencies cần thiết
- Tải dữ liệu từ Google Drive: `https://drive.google.com/file/d/1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI/view?usp=sharing`
- Giải nén và tạo cấu trúc thư mục
- Cài đặt `unrar` trên Linux nếu cần
- Kiểm tra tính toàn vẹn dữ liệu

### 4. (Tùy chọn) Cài đặt Dependencies thủ công
Nếu muốn cài đặt trước:
```bash
pip install -r requirements.txt
```

### 5. (Tùy chọn) Tải dữ liệu thủ công
Chỉ cần thiết nếu notebook không thể tải tự động:

1. Tải file từ: `https://drive.google.com/file/d/1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI/view?usp=sharing`
2. Giải nén vào thư mục gốc của project
3. Đảm bảo cấu trúc như sau:
```
data/
├── train/
│   ├── images/          # ISIC_*.jpg files
│   └── ground_truth/    # ISIC_*_segmentation.png files
├── val/
│   ├── images/
│   └── ground_truth/
└── test/
    ├── images/
    └── ground_truth/
```

### 6. (Tùy chọn) Kiểm tra Setup
Nếu muốn kiểm tra trước khi chạy notebook:
```bash
python -c "
import torch
print('✅ PyTorch imported successfully!')
print(f'🔥 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎮 GPU: {torch.cuda.get_device_name(0)}')
"
```

**Lưu ý**: Các thư viện khác sẽ được cài đặt tự động trong notebook.

## 📊 Kiểm tra Data

**Tự động**: Notebook sẽ tự động kiểm tra dữ liệu sau khi tải.

**Thủ công** (nếu cần):
```python
import os

def check_data_structure():
    base_path = "data"
    splits = ["train", "val", "test"]

    for split in splits:
        img_path = os.path.join(base_path, split, "images")
        mask_path = os.path.join(base_path, split, "ground_truth")

        if os.path.exists(img_path) and os.path.exists(mask_path):
            img_count = len([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            mask_count = len([f for f in os.listdir(mask_path) if f.endswith('.png')])
            print(f"✅ {split}: {img_count} images, {mask_count} masks")
        else:
            print(f"❌ {split}: Missing directories")

check_data_structure()
```

## 🐛 Troubleshooting

### Lỗi CUDA/GPU
```bash
# Kiểm tra CUDA version
nvidia-smi

# Cài đặt PyTorch với CUDA version phù hợp
# Ví dụ cho CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Lỗi Memory
- Giảm batch_size trong notebook (từ 8 xuống 4 hoặc 2)
- Sử dụng gradient accumulation
- Resize ảnh nhỏ hơn (từ 512x512 xuống 256x256)

### Lỗi tải dữ liệu trong Notebook
```bash
# Nếu notebook không thể tải tự động:
# 1. Kiểm tra kết nối internet
# 2. Chạy lại cell tải dữ liệu
# 3. Tải thủ công từ Google Drive và giải nén vào thư mục data/
```

### Lỗi giải nén RAR trên Linux
```bash
# Notebook sẽ tự động cài đặt, nhưng nếu cần cài thủ công:
sudo apt-get install unrar  # Ubuntu/Debian
sudo yum install unrar      # CentOS/RHEL
brew install unrar          # macOS

# Hoặc sử dụng Python package:
pip install rarfile
```

### Lỗi Import
```bash
# Notebook sẽ tự động cài đặt, nhưng nếu có lỗi:
# 1. Restart kernel trong Jupyter
# 2. Chạy lại cell cài đặt thư viện

# Hoặc cài đặt thủ công:
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

## 🎯 Quick Start

**Siêu đơn giản**:
1. Clone repository
2. Chạy notebook
3. Tất cả sẽ được tự động hóa!

**Sau khi notebook chạy xong**, bạn có thể:

1. **Test training nhanh**:
```python
# Trong notebook, set num_epochs=1 để test nhanh
segformer_history = train_model(
    segformer_model,
    train_loader,
    val_loader,
    num_epochs=1,  # Chỉ 1 epoch để test
    lr=1e-4
)
```

2. **Xem dữ liệu ngay**:
```python
# Cell visualization sẽ tự động chạy trong notebook
# Hiển thị samples từ dataset
```

3. **Load model đã train**:
```python
# Nếu bạn có model đã train sẵn trong folder models/
model = SegFormerModel()
model.load_state_dict(torch.load('models/segformer_model.pth', map_location='cpu'))
```

## 📝 Notes

- **First run**: Lần đầu chạy sẽ mất thời gian để:
  - Cài đặt dependencies (~5-10 phút)
  - Tải dữ liệu (~10-15 phút tùy tốc độ mạng)
  - Download pre-trained weights (~5 phút)
- **Data size**: ISIC dataset khoảng 3-5GB
- **Training time**:
  - SegFormer: ~2-3 giờ (15 epochs)
  - U-Net: ~1-2 giờ (15 epochs)
  - DeepLabV3+: ~3-4 giờ (15 epochs)
- **Total setup time**: ~30-45 phút cho lần đầu chạy

## 🆘 Cần hỗ trợ?

1. Kiểm tra [Issues](https://github.com/blanatole/CV_Master/issues) trên GitHub
2. Tạo issue mới với:
   - OS và Python version
   - Error message đầy đủ
   - Steps để reproduce lỗi

---

**Happy coding! 🎉**
