# Phân đoạn khối u trong ảnh y tế sử dụng Pre-trained Models

Dự án này thực hiện bài toán phân đoạn khối u da (skin lesion segmentation) sử dụng các pre-trained models từ Hugging Face Transformers và Timm trên dataset ISIC với các cải tiến hiện đại.

## 🚀 Cập nhật mới: Sử dụng Google Drive API

Project đã được cập nhật để sử dụng **Google Drive API** thay vì `gdown` để tải dữ liệu, mang lại nhiều ưu điểm:

### ✅ Ưu điểm của Google Drive API:
- **Bảo mật cao hơn**: Sử dụng Service Account thay vì public link
- **Ổn định hơn**: Không bị giới hạn tốc độ như `gdown`
- **Kiểm soát quyền truy cập**: Có thể quản lý ai được phép tải dữ liệu
- **Hiển thị tiến trình**: Thanh tiến trình chi tiết khi tải file lớn
- **Xử lý lỗi tốt hơn**: Thông báo lỗi rõ ràng và hướng dẫn khắc phục

### 📁 Cấu trúc file mới:
- `data_downloader.py`: Module chính để tải dữ liệu từ Google Drive API
- `download.py`: Script đơn giản để tải dataset
- `test_notebook_cell.py`: Script test để kiểm tra notebook cell
- `service_account.json`: File credentials cho Google Drive API (cần tạo)

## 🎯 Mục tiêu

- Phân đoạn chính xác vùng khối u da trong ảnh y tế
- So sánh hiệu suất của các pre-trained models khác nhau
- Đánh giá kết quả bằng **Dice Coefficient** và **Jaccard Index** (IoU)
- Tối ưu hóa training với **Learning Rate Scheduling** hiện đại
- Tổ chức models một cách chuyên nghiệp trong folder `models/`

## 📊 Dataset

**ISIC (International Skin Imaging Collaboration)**
- Training images: Khoảng 2000+ ảnh
- Validation images: Khoảng 150+ ảnh  
- Test images: Khoảng 600+ ảnh
- Format: JPG (images) và PNG (masks)

Cấu trúc thư mục:
```
data/
├── train/
│   ├── images/          # Ảnh training
│   └── ground_truth/    # Mask training
├── val/
│   ├── images/          # Ảnh validation
│   └── ground_truth/    # Mask validation
└── test/
    ├── images/          # Ảnh test
    └── ground_truth/    # Mask test
```

## 🚀 Models được sử dụng

### 1. SegFormer (Hugging Face)
- **Model**: `nvidia/segformer-b0-finetuned-ade-512-512`
- **Ưu điểm**: Hiệu suất cao, ít tham số (3.7M)
- **Kiến trúc**: Transformer-based segmentation
- **Learning Rate Scheduler**: Cosine Annealing (phù hợp với transformer)
- **Saved as**: `models/segformer_model.pth`

### 2. Vision Transformer (ViT) Segmentation
- **Model**: `google/vit-base-patch16-224` + Custom segmentation head
- **Ưu điểm**: Attention mechanism mạnh mẽ, hiệu suất cao trên medical images
- **Tham số**: ~86M
- **Kiến trúc**: Pure transformer với patch-based processing
- **Learning Rate Scheduler**: Cosine Annealing với warmup
- **Saved as**: `models/vit_segmentation_model.pth`

### 3. U-Net với EfficientNet Backbone (Timm)
- **Backbone**: EfficientNet-B0 (`efficientnet-b0`)
- **Ưu điểm**: Cân bằng tốt giữa hiệu suất và tốc độ
- **Tham số**: ~5.3M
- **Learning Rate Scheduler**: ReduceLROnPlateau (adaptive)
- **Saved as**: `models/unet_efficientnet_model.pth`

### 4. U-Net với ViT Backbone (Timm)
- **Backbone**: Vision Transformer Base (`vit_base_patch16_224`)
- **Ưu điểm**: Kết hợp ưu điểm của ViT và U-Net architecture
- **Tham số**: ~90M
- **Learning Rate Scheduler**: Cosine Annealing với warmup
- **Saved as**: `models/unet_vit_model.pth`

### 5. DeepLabV3+ với ResNet Backbone (Timm)
- **Backbone**: ResNet-50
- **Ưu điểm**: Robust, xử lý tốt multi-scale objects
- **Tham số**: ~39.6M
- **Learning Rate Scheduler**: StepLR (scheduled decay)
- **Saved as**: `models/deeplabv3_resnet_model.pth`

## 📋 Yêu cầu hệ thống

### Python packages:
```bash
# Core ML libraries
pip install transformers timm torch torchvision
pip install segmentation-models-pytorch accelerate datasets
pip install albumentations opencv-python-headless
pip install matplotlib seaborn scikit-learn pillow tqdm

# Google Drive API (mới)
pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2

# Hoặc cài đặt tất cả từ requirements.txt
pip install -r requirements.txt
```

### 🔑 Setup Google Drive API:

#### Bước 1: Tạo Service Account
1. Truy cập [Google Cloud Console](https://console.cloud.google.com/)
2. Tạo project mới hoặc chọn project hiện có
3. Bật Google Drive API:
   - Vào **APIs & Services** > **Library**
   - Tìm "Google Drive API" và bật nó
4. Tạo Service Account:
   - Vào **APIs & Services** > **Credentials**
   - Click **Create Credentials** > **Service Account**
   - Đặt tên và tạo service account
5. Tạo key cho Service Account:
   - Click vào service account vừa tạo
   - Vào tab **Keys** > **Add Key** > **Create New Key**
   - Chọn **JSON** và tải về
   - Đổi tên file thành `service_account.json`

#### Bước 2: Chia sẻ Google Drive file
1. Mở file Google Drive cần tải (dataset)
2. Click **Share** (Chia sẻ)
3. Thêm email của service account (có dạng `xxx@project-name.iam.gserviceaccount.com`)
4. Cấp quyền **Viewer** hoặc **Editor**

#### Bước 3: Đặt file credentials
```bash
# Đặt file service_account.json vào thư mục gốc của project
mv ~/Downloads/service_account.json ./service_account.json
```

### Hardware:
- **GPU**: NVIDIA GPU với ít nhất 8GB VRAM (khuyến nghị)
- **RAM**: Ít nhất 16GB
- **Storage**: Ít nhất 10GB cho dataset và models

## 🔧 Cách sử dụng

### 1. Clone repository
```bash
git clone https://github.com/blanatole/CV_Master.git
cd CV_Master
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Google Drive API (bắt buộc)
Làm theo hướng dẫn ở phần **🔑 Setup Google Drive API** ở trên để tạo `service_account.json`

### 4. Tải dữ liệu
✨ **Tính năng mới**: Sử dụng Google Drive API thay vì gdown!

**Cách 1: Sử dụng script download.py**
```bash
python download.py
```

**Cách 2: Sử dụng trong notebook**
Notebook sẽ tự động tải dữ liệu khi chạy cell tương ứng.

**Cách 3: Sử dụng module trực tiếp**
```python
from data_downloader import GoogleDriveDownloader, DatasetManager

# Khởi tạo downloader
downloader = GoogleDriveDownloader('service_account.json')
dataset_manager = DatasetManager(downloader)

# Tải dataset
file_id = "1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI"
dataset_manager.download_and_extract_dataset(file_id)
```

**Google Drive Link**: `https://drive.google.com/file/d/1IL3JPRaxhKoQMjPk_AzNK5w4OsE2gjsI/view?usp=sharing`

Hệ thống sẽ tự động:
- ✅ Xác thực với Google Drive API
- 📥 Tải dữ liệu với thanh tiến trình
- 📂 Giải nén file RAR/ZIP
- 📁 Tạo cấu trúc thư mục
- 🔧 Cài đặt `unrar` trên Linux nếu cần
- ✔️ Kiểm tra tính toàn vẹn dữ liệu

### 5. Chạy notebook
```bash
jupyter notebook medical_tumor_segmentation.ipynb
```

**📋 Lưu ý**:
- Notebook sẽ tự động cài đặt dependencies và tải dữ liệu
- Chạy các cell theo thứ tự từ trên xuống dưới
- Xem file `SETUP.md` để có hướng dẫn chi tiết thiết lập trên thiết bị mới
- Xem file `VIT_GUIDE.md` để hiểu chi tiết về Vision Transformer models

### 6. Training models
Notebook bao gồm code để train 3 models với **Learning Rate Scheduling**:
- **SegFormer**: 20 epochs với Cosine Annealing
- **U-Net với EfficientNet**: 25 epochs với ReduceLROnPlateau
- **DeepLabV3+ với ResNet**: 15 epochs với StepLR

Tất cả models được lưu trong folder `models/` để tổ chức tốt hơn.

### 6. Evaluation
Sử dụng các functions có sẵn để:
- Đánh giá performance trên test set với **Dice Coefficient** và **Jaccard Index**
- Visualize predictions từ tất cả models
- So sánh training histories và learning rate schedules
- Tạo performance comparison table

## 📈 Metrics đánh giá chính

### 🎯 **Dice Coefficient** (Primary Metric)
```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```
- **Ý nghĩa**: Đo độ tương tự giữa prediction và ground truth
- **Phạm vi**: 0-1 (càng cao càng tốt)
- **Ưu điểm**: Robust với class imbalance

### 🎯 **Jaccard Index (IoU)** (Secondary Metric)
```
Jaccard = |A ∩ B| / |A ∪ B|
```
- **Ý nghĩa**: Đo độ chồng lấp giữa prediction và ground truth
- **Phạm vi**: 0-1 (càng cao càng tốt)
- **Quan hệ**: Jaccard = Dice / (2 - Dice)

### 📊 **Combined Loss Function**
```
Combined Loss = α * BCE Loss + (1-α) * Dice Loss
```
- **α = 0.5**: Cân bằng giữa pixel-wise accuracy và shape similarity
- **Lợi ích**: Tối ưu hóa cả độ chính xác pixel và hình dạng tổng thể

### 📋 **Additional Metrics**
- **Sensitivity (Recall)**: `TP / (TP + FN)` - Tỷ lệ phát hiện đúng khối u
- **Specificity**: `TN / (TN + FP)` - Tỷ lệ loại bỏ đúng vùng không phải khối u

## 🎨 Data Augmentation

Sử dụng Albumentations với các transforms:
- **Geometric**: HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate
- **Color**: RandomBrightnessContrast, HueSaturationValue
- **Normalization**: ImageNet standards (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Resize**: 512x512 pixels cho tất cả models

## 🔧 Cải tiến kỹ thuật

### 📊 **Enhanced Metrics System**
- **Primary**: Dice Coefficient (robust với class imbalance)
- **Secondary**: Jaccard Index (IoU) cho comparison
- **Real-time tracking**: Train/Val metrics trong mỗi epoch
- **Comprehensive logging**: Loss, Dice, Jaccard, Learning Rate

### 📈 **Advanced Learning Rate Scheduling**
- **Model-specific schedulers**: Phù hợp với từng architecture
- **Automatic LR change detection**: Hiển thị khi LR thay đổi
- **Visualization**: Plot LR schedules để so sánh

### 💾 **Professional Model Management**
```
models/
├── segformer_model.pth          # SegFormer weights
├── unet_efficientnet_model.pth  # U-Net weights
└── deeplabv3_resnet_model.pth   # DeepLabV3+ weights
```

### 🎯 **Enhanced Loss Function**
- **Combined Loss**: Tối ưu cả pixel accuracy và shape similarity
- **Adaptive weighting**: α=0.5 cân bằng BCE và Dice Loss
- **Gradient stability**: Smooth training convergence

## 🔍 Kết quả mong đợi

| Model | Jaccard Index | Dice Coefficient | Parameters | LR Scheduler |
|-------|---------------|------------------|------------|--------------|
| SegFormer | ~0.82 | ~0.90 | 3.7M | Cosine Annealing |
| ViT Segmentation | ~0.84 | ~0.91 | 86M | Cosine + Warmup |
| U-Net (EfficientNet) | ~0.81 | ~0.89 | 5.3M | ReduceLROnPlateau |
| U-Net (ViT) | ~0.83 | ~0.90 | 90M | Cosine + Warmup |
| DeepLabV3+ (ResNet) | ~0.80 | ~0.89 | 39.6M | StepLR |

### 🎯 **Performance Highlights:**
- **SegFormer**: Tốt nhất về efficiency (ít tham số nhất)
- **U-Net**: Cân bằng tốt giữa performance và complexity
- **DeepLabV3+**: Robust nhất với multi-scale features

*Lưu ý: Kết quả có thể thay đổi tùy thuộc vào dataset và hyperparameters*

## 📝 Cấu trúc Notebook

1. **Import và cài đặt**: Cài đặt thư viện cần thiết (tự động)
2. **Tải dữ liệu**: Tự động tải và giải nén từ Google Drive
3. **Khám phá dữ liệu**: Phân tích dataset, visualize samples
4. **Dataset và Augmentation**: Tạo PyTorch Dataset và DataLoader (bao gồm test set)
5. **Model Definitions**: Định nghĩa các model architectures
   - 5.1 SegFormer (Hugging Face)
   - 5.2 U-Net với EfficientNet Backbone
   - 5.3 DeepLabV3+ với ResNet Backbone
6. **Loss Functions**: Dice Loss, Combined Loss (BCE + Dice), Metrics (Dice & Jaccard)
7. **Training Functions**: Enhanced training với comprehensive metrics tracking
8. **Training Models**:
   - 8.1 SegFormer với Cosine Annealing
   - 8.2 U-Net với ReduceLROnPlateau
   - 8.3 DeepLabV3+ với StepLR
9. **Visualization**: Plot training curves, learning rate schedules, so sánh models
10. **Evaluation**: Đánh giá comprehensive trên test set với Dice & Jaccard
11. **Demo**: Hướng dẫn sử dụng models đã train từ folder `models/`
12. **Performance Comparison**: Bảng so sánh chi tiết với actual results
13. **Kết luận**: Tổng kết cải tiến và hướng phát triển

## 🚨 Lưu ý quan trọng

### Preprocessing:
- Resize ảnh về 512x512 pixels
- Normalize theo ImageNet standards
- Binary threshold cho masks (>127 = 1, ≤127 = 0)

### Training:
- **Batch size**: 8 (có thể điều chỉnh theo GPU memory)
- **Learning rates**:
  - SegFormer: 5e-5 (lower for pre-trained transformer)
  - U-Net & DeepLabV3+: 1e-4
- **Optimizer**: Adam với weight decay 1e-4
- **Learning Rate Schedulers**:
  - **SegFormer**: Cosine Annealing (T_max=epochs, eta_min=lr*0.01)
  - **U-Net**: ReduceLROnPlateau (patience=3, factor=0.5)
  - **DeepLabV3+**: StepLR (step_size=epochs//3, gamma=0.1)
- **Loss Function**: Combined Loss (α=0.5) = BCE Loss + Dice Loss

### Validation:
- Sử dụng separate validation set
- **Metrics tracking**: Train/Val Loss, Dice, Jaccard, Learning Rate
- **Model saving**: Tất cả models lưu trong folder `models/`
- **Real-time monitoring**: Progress bars với metrics display

## 🔮 Hướng phát triển

### 🚀 **Đã thực hiện**:
- ✅ **Dice Coefficient & Jaccard Index** làm metrics chính
- ✅ **Learning Rate Scheduling** với 4 loại scheduler
- ✅ **Model Organization** trong folder `models/`
- ✅ **Enhanced Training Monitoring** với comprehensive metrics
- ✅ **Combined Loss Function** (BCE + Dice)

### 🎯 **Kế hoạch tiếp theo**:
1. **Advanced Augmentation**: MixUp, CutMix, Mosaic
2. **Ensemble Methods**: Kết hợp predictions từ nhiều models
3. **Post-processing**: CRF, Watershed segmentation
4. **Multi-scale Training**: Train với nhiều resolutions
5. **Transfer Learning**: Fine-tune từ medical datasets khác
6. **3D Segmentation**: Mở rộng cho volumetric data
7. **Model Optimization**: Pruning, Quantization, ONNX export
8. **Advanced Schedulers**: Warm-up, Polynomial decay

## 📚 Tài liệu tham khảo

- [SegFormer Paper](https://arxiv.org/abs/2105.15203)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [DeepLabV3+ Paper](https://arxiv.org/abs/1802.02611)
- [ISIC Dataset](https://www.isic-archive.com/)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

---

**Lưu ý**: Đây là project nghiên cứu/học tập. Không sử dụng cho mục đích chẩn đoán y tế thực tế mà không có sự giám sát của chuyên gia y tế.
