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

## 📚 **5 NOTEBOOKS ĐỘC LẬP** - Cách sử dụng mới

Project đã được chia thành **5 notebooks nhỏ** để dễ dàng quản lý và chạy từng phần riêng biệt:

### 📋 **Danh sách Notebooks:**

#### 1. 📥 `01_data_download.ipynb` - Tải và Chuẩn bị Dữ liệu
**Mục đích**: Tải dataset ISIC từ Google Drive và chuẩn bị dữ liệu cho training

**Nội dung**:
- Tải dữ liệu từ Google Drive API
- Giải nén và tạo cấu trúc thư mục
- Khám phá và visualize dữ liệu
- Phân tích thống kê dataset
- Tạo summary cho các notebook training

**Thời gian**: ~10-15 phút
**Yêu cầu**: File `service_account.json` cho Google Drive API

#### 2. 🤖 `02_train_segformer.ipynb` - Training SegFormer
**Mục đích**: Train model SegFormer (Transformer-based segmentation)

**Cấu hình**:
- **Model**: `nvidia/segformer-b0-finetuned-ade-512-512`
- **Parameters**: ~3.7M
- **Epochs**: 20
- **Learning Rate**: 5e-5
- **Scheduler**: Cosine Annealing
- **Ưu điểm**: Hiệu suất cao, ít tham số

**Thời gian**: ~2-3 giờ
**Output**: `models/segformer_model_*.pth`

#### 3. 🏗️ `03_train_unet_efficientnet.ipynb` - Training U-Net EfficientNet
**Mục đích**: Train U-Net với EfficientNet backbone

**Cấu hình**:
- **Backbone**: EfficientNet-B0
- **Parameters**: ~5.3M
- **Epochs**: 25
- **Learning Rate**: 1e-4
- **Scheduler**: ReduceLROnPlateau (adaptive)
- **Ưu điểm**: Cân bằng tốt giữa hiệu suất và tốc độ

**Thời gian**: ~1-2 giờ
**Output**: `models/unet_efficientnet_model_*.pth`

#### 4. 🔬 `04_train_unet_vit.ipynb` - Training U-Net ViT
**Mục đích**: Train U-Net với Vision Transformer backbone

**Cấu hình**:
- **Backbone**: Vision Transformer Base
- **Parameters**: ~90M
- **Epochs**: 20
- **Learning Rate**: 1e-4
- **Scheduler**: Cosine Annealing với Warmup (5 epochs)
- **Ưu điểm**: Attention mechanism mạnh mẽ

**Thời gian**: ~2-3 giờ
**Output**: `models/unet_vit_model_*.pth`

#### 5. 🎯 `05_train_deeplabv3_resnet.ipynb` - Training DeepLabV3+ ResNet
**Mục đích**: Train DeepLabV3+ với ResNet backbone

**Cấu hình**:
- **Backbone**: ResNet-50
- **Parameters**: ~39.6M
- **Epochs**: 15
- **Learning Rate**: 1e-4
- **Scheduler**: StepLR (step_size=5, gamma=0.1)
- **Ưu điểm**: Robust, xử lý tốt multi-scale objects

**Thời gian**: ~3-4 giờ
**Output**: `models/deeplabv3_resnet_model_*.pth`

### 🚀 **Cách chạy Notebooks:**

#### **Bước 1: Chuẩn bị môi trường**
```bash
# Clone repository
git clone <repository_url>
cd cv_huit

# Cài đặt dependencies
pip install -r requirements.txt
```

#### **Bước 2: Setup Google Drive API (cho notebook 1)**
1. Tạo Service Account trên Google Cloud Console
2. Tải file credentials và đổi tên thành `service_account.json`
3. Chia sẻ dataset với service account email

#### **Bước 3: Chạy notebooks theo thứ tự**

##### 📥 **Bắt buộc**: Chạy notebook 1 trước
```bash
jupyter notebook 01_data_download.ipynb
```
- Chạy tất cả cells từ trên xuống dưới
- Đảm bảo dữ liệu được tải thành công

##### 🤖 **Tùy chọn**: Chạy các notebook training (2-5)
Bạn có thể chạy **bất kỳ notebook nào** trong số 4 notebook training:

```bash
# SegFormer (nhẹ nhất, nhanh nhất)
jupyter notebook 02_train_segformer.ipynb

# U-Net EfficientNet (cân bằng)
jupyter notebook 03_train_unet_efficientnet.ipynb

# U-Net ViT (hiệu suất cao)
jupyter notebook 04_train_unet_vit.ipynb

# DeepLabV3+ ResNet (robust nhất)
jupyter notebook 05_train_deeplabv3_resnet.ipynb
```

### 📊 **So sánh Models**

| Model | Parameters | Training Time | Dice Score* | Jaccard Index* | Scheduler |
|-------|------------|---------------|-------------|----------------|-----------|
| SegFormer | 3.7M | 2-3h | ~0.90 | ~0.82 | Cosine Annealing |
| U-Net EfficientNet | 5.3M | 1-2h | ~0.89 | ~0.81 | ReduceLROnPlateau |
| U-Net ViT | 90M | 2-3h | ~0.90 | ~0.83 | Cosine + Warmup |
| DeepLabV3+ ResNet | 39.6M | 3-4h | ~0.89 | ~0.80 | StepLR |

*Kết quả dự kiến, có thể thay đổi tùy thuộc vào dataset và hyperparameters

### 💾 **Cấu trúc Output**

Sau khi chạy các notebooks, bạn sẽ có:

```
cv_huit/
├── data/                           # Dataset (từ notebook 1)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                         # Trained models (từ notebooks 2-5)
│   ├── segformer_model_best.pth
│   ├── segformer_model_final.pth
│   ├── unet_efficientnet_model_best.pth
│   ├── unet_efficientnet_model_final.pth
│   ├── unet_vit_model_best.pth
│   ├── unet_vit_model_final.pth
│   ├── deeplabv3_resnet_model_best.pth
│   └── deeplabv3_resnet_model_final.pth
└── notebooks/
    ├── 01_data_download.ipynb
    ├── 02_train_segformer.ipynb
    ├── 03_train_unet_efficientnet.ipynb
    ├── 04_train_unet_vit.ipynb
    └── 05_train_deeplabv3_resnet.ipynb
```

### 🎯 **Lựa chọn Model phù hợp**

#### 🚀 **Nếu bạn muốn nhanh và hiệu quả**:
→ Chạy `02_train_segformer.ipynb` (SegFormer)
- Ít tham số nhất (3.7M)
- Hiệu suất cao
- Training nhanh

#### ⚖️ **Nếu bạn muốn cân bằng**:
→ Chạy `03_train_unet_efficientnet.ipynb` (U-Net EfficientNet)
- Cân bằng tốt giữa hiệu suất và tốc độ
- Training nhanh nhất (1-2h)

#### 🎯 **Nếu bạn muốn hiệu suất cao nhất**:
→ Chạy `04_train_unet_vit.ipynb` (U-Net ViT)
- Attention mechanism mạnh mẽ
- Hiệu suất cao trên medical images

#### 🛡️ **Nếu bạn muốn robust nhất**:
→ Chạy `05_train_deeplabv3_resnet.ipynb` (DeepLabV3+ ResNet)
- Xử lý tốt multi-scale objects
- ASPP module cho multi-scale features

### 🔧 **Troubleshooting**

#### ❌ **Lỗi Google Drive API**:
- Kiểm tra file `service_account.json`
- Đảm bảo service account có quyền truy cập file
- Kiểm tra kết nối internet

#### ❌ **Lỗi giải nén RAR** (phổ biến nhất):
```bash
# Chạy script fix tự động
python fix_extraction.py

# Hoặc cài đặt thủ công
pip install rarfile

# Linux/Ubuntu
sudo apt install unrar

# macOS
brew install unrar

# Windows: Tải WinRAR hoặc 7-Zip
```

#### ❌ **Lỗi GPU Memory**:
- Giảm batch_size từ 8 xuống 4 hoặc 2
- Sử dụng CPU nếu cần: `device = 'cpu'`

#### ❌ **Lỗi Dependencies**:
```bash
pip install --upgrade torch torchvision
pip install --upgrade transformers timm
pip install segmentation-models-pytorch
```

### 📝 **Notes**

- **Notebook 1** là **bắt buộc** phải chạy trước
- **Notebooks 2-5** có thể chạy **độc lập** với nhau
- Mỗi notebook tự động lưu model tốt nhất và model cuối cùng
- Tất cả notebooks đều có visualization và evaluation
- Có thể dừng và tiếp tục training bằng cách load model đã lưu

### 🎉 **Kết luận**

Project đã được tổ chức thành **5 notebooks độc lập** để:
- ✅ Dễ dàng quản lý và debug
- ✅ Có thể chạy từng model riêng biệt
- ✅ Tiết kiệm thời gian khi chỉ cần train 1 model
- ✅ Dễ dàng so sánh và thử nghiệm
- ✅ Có thể chạy song song trên nhiều GPU/máy khác nhau

**Happy Training! 🚀**

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

## 📝 Cấu trúc Notebooks

### 📥 **Notebook 1: Data Download**
1. **Import thư viện**: Google Drive API, visualization tools
2. **Google Drive Setup**: Xác thực và kết nối API
3. **Download Dataset**: Tải ISIC dataset với progress bar
4. **Data Exploration**: Phân tích cấu trúc, visualize samples
5. **Statistics**: Thống kê kích thước, phân bố mask
6. **Summary**: Chuẩn bị thông tin cho training notebooks

### 🤖 **Notebook 2: SegFormer Training**
1. **Setup**: Import transformers, segmentation libraries
2. **Dataset**: ISIC dataset với augmentation
3. **Model**: SegFormer từ Hugging Face
4. **Training**: Cosine Annealing scheduler, 20 epochs
5. **Evaluation**: Dice & Jaccard metrics, visualization
6. **Results**: Performance summary và model saving

### 🏗️ **Notebook 3: U-Net EfficientNet Training**
1. **Setup**: Segmentation models pytorch, timm
2. **Dataset**: Cùng dataset với augmentation khác nhau
3. **Model**: U-Net với EfficientNet-B0 backbone
4. **Training**: ReduceLROnPlateau scheduler, 25 epochs
5. **Evaluation**: Comprehensive metrics tracking
6. **Results**: Comparison với SegFormer

### 🔬 **Notebook 4: U-Net ViT Training**
1. **Setup**: Vision Transformer libraries, einops
2. **Dataset**: Optimized cho ViT input requirements
3. **Model**: U-Net với ViT backbone (hoặc ResNet fallback)
4. **Training**: Cosine Annealing với Warmup, 20 epochs
5. **Evaluation**: Advanced metrics và attention visualization
6. **Results**: High-performance model analysis

### 🎯 **Notebook 5: DeepLabV3+ ResNet Training**
1. **Setup**: Multi-scale segmentation setup
2. **Dataset**: Same preprocessing pipeline
3. **Model**: DeepLabV3+ với ResNet-50, ASPP module
4. **Training**: StepLR scheduler, 15 epochs
5. **Evaluation**: Multi-scale performance analysis
6. **Results**: Robust model comparison và final summary

### 📊 **Tổng kết tất cả Notebooks**
- **Notebook 1**: Bắt buộc chạy trước (data preparation)
- **Notebooks 2-5**: Độc lập, có thể chạy bất kỳ thứ tự nào
- **Mỗi notebook**: Complete pipeline từ setup đến evaluation
- **Output**: Trained models trong folder `models/`
- **Visualization**: Training curves, predictions, comparisons

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
