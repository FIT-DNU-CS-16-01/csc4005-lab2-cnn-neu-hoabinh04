# CSC4005 – Lab 2 Report: CNN for NEU Surface Defect Classification

## 1. Thông tin chung
- Họ và tên: Hoà Bình đẹp zai
- Lớp: KHMT 16-01
- Repo: csc4005-lab2-cnn-neu-hoabinh04
- W&B project: csc4005-lab2-neu-cnn

## 2. Bài toán
**Phân loại 6 lớp khiếm khuyết bề mặt NEU (NEU-CLS Dataset)**

Bài toán phân loại ảnh công nghiệp 6 lớp: Crazing, Inclusion, Patches, Pitted_Surface, Rolled-in_Scale, Scratches. Dataset có 1200 ảnh chia thành train:validation:test = 60:20:20 (test set: 270 mẫu).

**Mục tiêu Lab 2**: So sánh hiệu suất giữa:
1. CNN từ đầu (scratch) vs MLP baseline từ Lab 1
2. Transfer learning (pretrained backbone) vs training from scratch
3. Finetune vs Transfer learning (frozen backbone)

## 3. Mô hình và cấu hình

### 3.1. MLP baseline từ Lab 1
- Architecture: 2 hidden layers (512, 256 units), ReLU activation, Dropout 0.3
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Epochs: 20, Batch size: 32
- Test Accuracy: **0.3593** (35.93%)
- Trainable Params: ~650K

### 3.2. CNN Small from Scratch (Baseline)
- Architecture: 4 Conv layers (16→32→64→128 filters), 3x3 kernels, MaxPooling, 2 FC layers
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
- Epochs: 20, Batch size: 32, Image size: 64×64
- Early stopping: patience=5
- Best Val Acc: 0.9556 (95.56%)
- Test Accuracy: **0.9556** (95.56%)
- Trainable Params: 32,614

### 3.3. ResNet18 Transfer Learning
- Architecture: ResNet18 pretrained on ImageNet, frozen backbone + new classifier
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Epochs: 10, Batch size: 32, Image size: 128×128
- Early stopping: patience=3
- Best Val Acc: 0.9667 (96.67%)
- Test Accuracy: **0.9630** (96.30%)
- Trainable Params: 3,078 (only classifier head)

### 3.4. ResNet18 Finetune (BEST MODEL ⭐)
- Architecture: ResNet18 pretrained on ImageNet, all layers unfrozen
- Optimizer: AdamW (lr=0.0001, weight_decay=0.0001) - lower LR for finetune
- Epochs: 10, Batch size: 32, Image size: 128×128
- Early stopping: patience=3
- Best Val Acc: 1.0000 (perfect!)
- Test Accuracy: **0.9963** (99.63%) ⭐
- Trainable Params: 11,179,590 (all backbone layers)

## 4. Bảng kết quả so sánh

| Model | Train Mode | Best Val Acc | Test Acc | Epoch Time (s) | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP | scratch | 0.4037 | 0.3593 | ~3 | 650K | Baseline so sánh (Lab 1) |
| CNN-small | scratch | 0.9556 | 0.9556 | 4.2 | 32K | Tốt hơn MLP rất nhiều |
| ResNet18 | transfer | 0.9667 | 0.9630 | 14.8 | 3K | Ít parameter hơn |
| **ResNet18** | **finetune** | **1.0000** | **0.9963** | 35.7 | 11.2M | **BEST - Perfect validation!** |

## 5. Phân tích learning curves

### 5.1. CNN Small (Scratch)
- Train loss giảm đều: 1.399 → 0.162 (epoch 1→20)
- Val loss: giảm rồi ổn định ở ~0.14
- Train acc: 0.526 → 0.942
- Val acc: 0.244 → 0.941 (peak 0.956 epoch 18)
- **Nhận xét**: Learning curve ổn định, không overfitting rõ rệt. Val acc bắt kịp train acc.

### 5.2. ResNet18 Transfer Learning
- Train loss: 1.347 → 0.227 (epoch 1→10)
- Val loss: 0.937 → 0.151 (giảm liên tục)
- Train acc: 0.509 → 0.930
- Val acc: 0.796 → 0.967
- **Nhận xét**: Transfer learning học rất nhanh. Epoch 1 đã đạt 79.6% val acc nhờ pretrained backbone.

### 5.3. ResNet18 Finetune (BEST)
- Epoch 1: val_acc=0.9556 (cực tốt nhờ pretrained feature)
- Epoch 2: val_acc=0.9926 (98.26%)
- Epoch 4: val_acc=1.0000 (perfect!)
- Epoch 7-10: val_acc=1.0 hoặc 0.996 (rất ổn định)
- Train loss: 0.498 → 0.024 (giảm rất nhanh)
- Val loss: 0.135 → 0.008 (xuống mức rất thấp)
- **Nhận xét**: Finetune hội tụ siêu nhanh chỉ trong 4 epoch. Đạt perfect validation accuracy.

## 6. Confusion matrix và lỗi dự đoán

### 6.1. ResNet18 Finetune (Best Model)
```
Confusion Matrix (270 test samples):
                  Predicted
                  Cra  Inc  Pat  Pit  Rol  Scr
Actual  Crazing     44    0    0    1    0    0   (1 error: Crazing→Pitted_Surface)
        Inclusion    0   45    0    0    0    0   (0 errors)
        Patches      0    0   45    0    0    0   (0 errors)
        Pitted_Surf  0    0    0   45    0    0   (0 errors)
        Rolled-in    0    0    0    0   45    0   (0 errors)
        Scratches    0    0    0    0    0   45   (0 errors)
```

**Per-class metrics:**
- Crazing: precision=1.0, recall=0.9778, f1=0.9888 (44/45 đúng)
- Inclusion: precision=1.0, recall=1.0, f1=1.0 (45/45 đúng)
- Patches: precision=1.0, recall=1.0, f1=1.0 (45/45 đúng)
- Pitted_Surface: precision=0.978, recall=1.0, f1=0.989 (45/45 đúng)
- Rolled-in_Scale: precision=1.0, recall=1.0, f1=1.0 (45/45 đúng)
- Scratches: precision=1.0, recall=1.0, f1=1.0 (45/45 đúng)

**Macro avg: precision=0.996, recall=0.996, f1=0.996**

### 6.2. Phân tích lỗi
- **Tổng sai: 1/270** (0.37% error rate)
- **Lỗi duy nhất**: 1 mẫu Crazing được dự đoán là Pitted_Surface
- **Nguyên nhân**: Có thể do overlap visual features giữa Crazing (mạng nứt) và Pitted_Surface (lỗ hổng)
- **So sánh**:
  - MLP: 173/270 lỗi (64% error)
  - CNN-small: 12/270 lỗi (4.4% error)
  - ResNet18 Transfer: 10/270 lỗi (3.7% error)
  - ResNet18 Finetune: **1/270 lỗi (0.37% error)** ⭐

## 7. Kết luận

### 7.1. CNN có cải thiện so với MLP không?
**CÓ, cải thiện rất đáng kể:**
- MLP: test_acc = 0.3593 (35.93%)
- CNN-small: test_acc = 0.9556 (95.56%)
- **Cải thiện: +60 percentage points** 🚀

**Lý do**: CNN giữ cấu trúc không gian của ảnh (spatial structure) thông qua convolutional layers, trong khi MLP coi ảnh là một dãy số flat. CNN học được local features (edges, textures) qua kernel sliding, weight sharing giảm parameters từ 650K xuống 32K.

### 7.2. Transfer learning có tốt hơn không?
**CÓ, nhưng với trade-off khác nhau:**

**Transfer Learning (Frozen Backbone):**
- Test acc: 0.9630 (96.30%)
- Epoch time: 14.8s (nhanh hơn CNN-small 3.5x do larger backbone)
- Trainable params: 3,078 (cực ít, chỉ classifier)
- **Ưu điểm**: Nhanh train, parameters ít, tốt nếu data limited
- **Nhược điểm**: Một số features từ pretrained model có thể không phù hợp 100%

**Finetune (Unfrozen Backbone):**
- Test acc: 0.9963 (99.63%) - BEST ⭐
- Epoch time: 35.7s (chậm hơn do update tất cả parameters)
- Trainable params: 11.2M (full model)
- **Ưu điểm**: Độ chính xác cao nhất, adapt hoàn toàn với task
- **Nhược điểm**: Chậm hơn, risk overfitting nếu data ít

### 7.3. Khi nào chọn Transfer Learning vs Train from Scratch?

| Tiêu chí | From Scratch | Transfer | Finetune |
|---|---|---|---|
| **Data availability** | Nhiều | Ít-trung bình | Trung bình |
| **Training time** | Lâu | Nhanh | Chậm nhất |
| **Accuracy** | Trung bình-tốt | Tốt | Tốt nhất |
| **Parameters** | Ít | Cực ít | Nhiều |
| **Computational cost** | Vừa | Thấp | Cao |
| **Overfitting risk** | Vừa | Thấp | Cao (nếu data ít) |
| **Lab này phù hợp** | CNN-small ✓ | ResNet transfer ✓ | ResNet finetune ✅ BEST |

**Khuyến nghị:**
- **Transfer (frozen)**: Khi data < 1000 mẫu hoặc time/compute bị giới hạn
- **Finetune**: Khi data ≥ 1000 mẫu và có GPU mạnh, muốn accuracy cực cao
- **From scratch**: Khi problem domain rất khác ImageNet, hoặc muốn learn features từ đầu

### 7.4. Tại sao ResNet18 Finetune vượt trội?
1. **Pretrained initialization**: ImageNet features rất phong phú, phù hợp với vision tasks
2. **Unfrozen backbone**: Adapt tất cả layers cho NEU-CLS dataset
3. **Small learning rate (0.0001)**: Giữ pretrained weights, chỉ fine-tune nhẹ nhàng
4. **Augmentation**: Data augmentation giúp generalize tốt hơn
5. **Architecture**: ResNet với skip connections học deep better

### 7.5. Benchmark Final
```
Model Performance Ranking:
1. ResNet18 Finetune:    99.63% ⭐⭐⭐ (BEST - only 1 error in 270)
2. ResNet18 Transfer:    96.30% ⭐⭐  (Good, 10 errors)
3. CNN-small Scratch:    95.56% ⭐   (Solid, 12 errors)
4. MLP Baseline (Lab 1): 35.93%     (Poor, 173 errors)
```

**W&B Dashboard**: https://wandb.ai/1671040004-dai-nam/csc4005-lab2-neu-cnn
(Các learning curves được log ở chế độ offline, có thể sync khi setup wandb hoàn toàn)
