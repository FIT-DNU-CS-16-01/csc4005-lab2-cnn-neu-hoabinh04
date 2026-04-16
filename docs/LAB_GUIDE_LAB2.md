# Lab 2 – CNN Image Classification (From Scratch vs Transfer)  
## CSC4005 – Bài 3: Mạng neuron tích chập - CNN

## 1. Thông tin lab
- **Học phần:** CSC4005 – Học sâu  
- **Bài học liên quan:** **Bài 3 – Mạng neuron tích chập - CNN**  
- **Tên lab:** **Lab 2 – CNN Image Classification (From Scratch vs Transfer)**  
- **Tuần:** W03  
- **Mức độ:** L2  
- **Case study:** Phân loại lỗi bề mặt thép trên **NEU Surface Defect Database / NEU-CLS.zip**  
- **Công cụ theo dõi thí nghiệm:** **Weights & Biases (W&B)**  

Lab này nối tiếp trực tiếp từ **Lab 1 – Training & Regularization**. Ở Lab 1, sinh viên đã làm việc với **MLP** để hiểu rõ pipeline huấn luyện, loss function, optimizer, regularization và experiment tracking. Sang Lab 2, sinh viên vẫn dùng **đúng tập dữ liệu NEU-CLS**, nhưng chuyển sang **CNN** để khai thác tốt hơn cấu trúc không gian của ảnh. Đồng thời, sinh viên tiếp tục sử dụng **W&B** để so sánh thí nghiệm một cách có căn cứ.

---

## 2. Căn chỉnh với bài học trong Notion
Lab này bám đúng trọng tâm của **Bài 3 – Mạng neuron tích chập - CNN**:  
- kiến trúc CNN, convolution, pooling, transfer learning mindset,  
- quick check về **receptive field**, **weight sharing**, và **khi nào chọn transfer learning hay train from scratch** 

Nó cũng bám đúng mô tả của **Lab 2 – CNN Image Classification (From Scratch vs Transfer)** trong Notion:  
- có **2 mô hình**: (A) CNN from scratch, (B) transfer learning,  
- có **bảng so sánh metrics + thời gian train/epoch**,  
- có kết luận **khi nào transfer learning tốt hơn** dựa trên số liệu.

---

## 3. Bối cảnh bài toán
Trong dây chuyền kiểm định bề mặt thép, mô hình cần phân loại ảnh grayscale vào một trong 6 loại lỗi:
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

Ở Lab 1, sinh viên đã dùng MLP và phải **flatten** ảnh thành vector 1 chiều. Cách đó giúp học rõ quy trình training, nhưng làm mất thông tin không gian của ảnh.  
Ở Lab 2, CNN sẽ xử lý ảnh theo **ma trận 2 chiều**, tận dụng:
- local patterns,
- weight sharing,
- receptive field tăng dần qua các lớp convolution/pooling,
- khả năng học đặc trưng cạnh, vân bề mặt, vùng lỗi tốt hơn MLP.

---

## 4. Mục tiêu học tập
Sau khi hoàn thành lab này, sinh viên cần làm được:

1. Giải thích vì sao **CNN phù hợp hơn MLP cho bài toán ảnh**.  
2. Xây dựng được một **CNN from scratch** cho phân loại 6 lớp.  
3. Dùng được một mô hình **transfer learning** đơn giản, ví dụ ResNet18 / VGG11 / MobileNetV2.  
4. Thực hiện so sánh có kiểm soát giữa:
   - MLP baseline từ Lab 1,
   - CNN from scratch,
   - Transfer learning.  
5. Dùng **W&B** để log đầy đủ cấu hình, learning curves, confusion matrix, thời gian train và kết quả cuối.  
6. Rút ra kết luận: **khi nào CNN from scratch đủ tốt, khi nào transfer learning đáng dùng hơn**.

---

## 5. Liên hệ với Lab 1
Lab 2 không bắt đầu từ đầu. Sinh viên phải kế thừa các nguyên tắc đã học ở Lab 1:
- không dùng test set để chọn mô hình,
- có validation set đúng nghĩa,
- dùng learning curves để đọc overfitting / underfitting,
- so sánh cấu hình có kiểm soát,
- log thí nghiệm bằng W&B.

Điểm mới của Lab 2 là:
- thay **MLP** bằng **CNN**,  
- học cách thiết kế stack **Conv → ReLU → Pool**,  
- tiếp cận **transfer learning**,
- phân tích khác biệt về **độ chính xác**, **tốc độ học**, **thời gian train**, **khả năng tổng quát hóa**.

---

## 6. Dữ liệu sử dụng
Sinh viên tiếp tục dùng đúng dữ liệu của Lab 1:
- file **`NEU-CLS.zip`**
- hoặc thư mục đã giải nén từ file này.

### Yêu cầu bắt buộc
- Không tạo thư mục `data/` trong repo starter kit.
- Khi chạy code, phải truyền dữ liệu qua tham số:

```bash
--data_dir /duong_dan/NEU-CLS.zip
```

### Quy ước chung
- tiếp tục dùng cùng cách split train / validation / test như Lab 1,
- tiếp tục dùng ảnh grayscale của NEU,
- có thể resize về `64x64`, `96x96` hoặc `128x128` tùy mô hình,
- với transfer learning, cần chuyển grayscale thành 3 channels nếu backbone pretrained yêu cầu đầu vào RGB.

---

## 7. W&B trong Lab 2
W&B vẫn là phần **bắt buộc**.

### Vì sao vẫn phải dùng W&B?
Vì Lab 2 không chỉ yêu cầu “train được CNN”, mà còn yêu cầu:
- so sánh nhiều mô hình,
- theo dõi learning curves,
- ghi lại thời gian train,
- lưu best validation metrics,
- so sánh from scratch với transfer learning bằng số liệu rõ ràng.

### Project name thống nhất
Dùng tên project W&B:

```text
csc4005-lab2-neu-cnn
```

### Mỗi run cần log tối thiểu
- `train_loss`
- `val_loss`
- `train_acc`
- `val_acc`
- `best_val_acc`
- `epoch_time_sec`
- `optimizer`
- `lr`
- `weight_decay`
- `dropout`
- `model_name`
- `train_mode` = `scratch` hoặc `transfer`

### Nên log thêm
- confusion matrix cho best model,
- số lượng tham số trainable,
- ảnh dự đoán đúng / sai tiêu biểu,
- test accuracy cuối cùng.

---

## 8. Nhiệm vụ thực hành

## Phần A – Baseline nhắc lại từ Lab 1
Sinh viên chọn lại **best MLP run** từ Lab 1 làm mốc tham chiếu.

### Yêu cầu
1. Ghi lại các chỉ số của best MLP:
   - best val accuracy,
   - test accuracy,
   - số epoch,
   - thời gian train trung bình mỗi epoch.
2. Tạo một bảng nhỏ trong báo cáo để dùng làm baseline so sánh với CNN.

> Mục đích: không chỉ biết CNN chạy được, mà còn biết CNN **cải thiện được gì so với MLP**.

---

## Phần B – CNN from scratch
Sinh viên xây dựng một CNN cơ bản từ đầu.

### Gợi ý kiến trúc tối thiểu
```text
Input (1 x H x W)
→ Conv(1, 16, kernel=3, padding=1) + ReLU
→ MaxPool(2)
→ Conv(16, 32, kernel=3, padding=1) + ReLU
→ MaxPool(2)
→ Conv(32, 64, kernel=3, padding=1) + ReLU
→ AdaptiveAvgPool / Flatten
→ FC(64 → 128) + ReLU + Dropout
→ FC(128 → 6)
```

### Yêu cầu bắt buộc
- dùng **CrossEntropyLoss**,
- dùng tối thiểu 1 optimizer trong số:
  - AdamW,
  - SGD,
- có validation set,
- có early stopping,
- có W&B logging,
- lưu best model theo validation.

### Câu hỏi sinh viên cần tự trả lời khi train CNN from scratch
1. CNN có hội tụ nhanh hơn MLP không?  
2. CNN có giảm overfitting tốt hơn MLP không?  
3. Các feature map đầu mạng có học được cấu trúc cục bộ rõ hơn không?

---

## Phần C – Transfer Learning
Sinh viên dùng một backbone pretrained để làm phân loại 6 lớp trên cùng bộ dữ liệu.

### Gợi ý backbone
- ResNet18
- VGG11
- MobileNetV2

### Cách làm gợi ý
#### Cách 1 – Freeze backbone trước
- giữ nguyên phần lớn backbone pretrained,
- chỉ thay classifier cuối thành 6 lớp,
- train classifier head trong vài epoch đầu.

#### Cách 2 – Fine-tune một phần
- sau khi head học ổn định, unfreeze thêm block cuối,
- fine-tune với learning rate nhỏ hơn.

### Lưu ý quan trọng
- nếu backbone pretrained từ ImageNet, cần chuẩn hóa input theo chuẩn backbone,
- cần đổi ảnh grayscale thành 3 channels,
- tránh augmentation quá mạnh làm sai đặc trưng lỗi bề mặt.

### Yêu cầu tối thiểu
Sinh viên phải có **ít nhất 1 run transfer learning** và mô tả rõ:
- backbone nào được dùng,
- freeze hay fine-tune,
- learning rate,
- batch size,
- image size,
- val/test performance.

---

## Phần D – So sánh có kiểm soát
Sinh viên phải có tối thiểu 3 mô hình / 3 nhóm run để so sánh:

1. **MLP baseline** từ Lab 1  
2. **CNN from scratch**  
3. **Transfer learning**  

### Bảng so sánh bắt buộc
| Model | Train mode | Best Val Acc | Test Acc | Epoch time | Params trainable | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP | scratch | ... | ... | ... | ... | baseline Lab 1 |
| CNN-small | scratch | ... | ... | ... | ... | học đặc trưng cục bộ tốt hơn |
| ResNet18 | transfer | ... | ... | ... | ... | hội tụ nhanh hơn / tốt hơn |

### Sinh viên cần rút ra kết luận
- CNN có cải thiện so với MLP không?  
- Transfer learning có luôn tốt hơn CNN from scratch không?  
- Khi dữ liệu không quá lớn, pretrained backbone có lợi gì?  
- Có tình huống nào from scratch vẫn là lựa chọn hợp lý?

---

## 9. Gợi ý cấu hình thí nghiệm
### Run 1 – CNN from scratch baseline
- `model_name = cnn_small`
- `optimizer = adamw`
- `lr = 1e-3`
- `weight_decay = 1e-4`
- `dropout = 0.3`
- `img_size = 64`
- `batch_size = 32`

### Run 2 – CNN from scratch regularized hơn
- `model_name = cnn_small`
- `optimizer = adamw`
- `lr = 5e-4`
- `weight_decay = 1e-3`
- `dropout = 0.5`
- `img_size = 64`
- `batch_size = 32`

### Run 3 – Transfer learning baseline
- `model_name = resnet18`
- `train_mode = transfer`
- `freeze_backbone = true`
- `lr = 1e-3`
- `img_size = 128`
- `batch_size = 32`

### Run 4 – Fine-tune block cuối
- `model_name = resnet18`
- `train_mode = finetune`
- `freeze_backbone = false`
- `lr = 1e-4`
- `img_size = 128`
- `batch_size = 32`

Sinh viên không bắt buộc phải dùng đúng các cấu hình trên, nhưng phải có logic so sánh rõ ràng.

---

## 10. Gợi ý lệnh chạy
### CNN from scratch
```bash
python -m src.train_cnn \
  --data_dir /duong_dan/NEU-CLS.zip \
  --project csc4005-lab2-neu-cnn \
  --run_name cnn_scratch_adamw \
  --model_name cnn_small \
  --optimizer adamw \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --dropout 0.3 \
  --epochs 20 \
  --batch_size 32 \
  --img_size 64 \
  --patience 5 \
  --augment \
  --use_wandb
```

### Transfer learning
```bash
python -m src.train_cnn \
  --data_dir /duong_dan/NEU-CLS.zip \
  --project csc4005-lab2-neu-cnn \
  --run_name resnet18_transfer \
  --model_name resnet18 \
  --train_mode transfer \
  --freeze_backbone \
  --optimizer adamw \
  --lr 0.001 \
  --epochs 15 \
  --batch_size 32 \
  --img_size 128 \
  --patience 4 \
  --augment \
  --use_wandb
```

> Nếu starter kit Lab 2 dùng file train khác, giảng viên chỉ cần giữ nguyên tinh thần: một lệnh cho CNN from scratch, một lệnh cho transfer learning, đều truyền `--data_dir` từ file ZIP.

---

## 11. Những lỗi rất hay gặp
Các lỗi này bám đúng phần common pitfalls trong Notion và được mở rộng cho đúng dataset hiện tại fileciteturn20file0L1-L1:

1. **Không freeze/unfreeze đúng cách**  
   → transfer learning không hiệu quả hoặc train quá chậm.

2. **Chuẩn hóa input sai chuẩn backbone pretrained**  
   → mô hình pretrained hoạt động kém bất thường.

3. **Augmentation quá mạnh**  
   → làm biến dạng đặc trưng lỗi bề mặt, dẫn tới label noise.

4. **Resize không hợp lý**  
   → ảnh quá nhỏ làm mất chi tiết lỗi, ảnh quá lớn gây nặng bộ nhớ.

5. **Dùng test để chọn mô hình**  
   → sai quy trình thực nghiệm.

6. **Chỉ nhìn accuracy mà bỏ qua learning curves**  
   → bỏ sót dấu hiệu overfitting.

---

## 12. Câu hỏi thảo luận
Các câu hỏi này bám đúng quick check của Bài 3 và được gắn với lab:

1. **Receptive field** tăng như thế nào khi chồng nhiều lớp conv/pooling?  
2. Vì sao **weight sharing** giúp CNN hiệu quả hơn MLP cho dữ liệu ảnh?  
3. Với bộ dữ liệu NEU-CLS, khi nào bạn ưu tiên **transfer learning** hơn **train from scratch**?  
4. Tại sao CNN có thể học tốt hơn MLP dù cả hai đều là supervised classification?  
5. Nếu transfer learning cho val accuracy cao hơn nhưng train chậm hơn, bạn đánh giá lựa chọn đó thế nào?

---

## 13. Deliverables
Phần nộp bài cần bám đúng tinh thần Lab 2 trong Notion:

1. **Hai mô hình chính**:
   - (A) CNN from scratch
   - (B) Transfer learning  
2. **Bảng so sánh metrics + thời gian train/epoch**  
3. **Learning curves** cho các run chính  
4. **Ảnh chụp hoặc link W&B project**  
5. **Kết luận bằng số liệu**: khi nào transfer learning tốt hơn  
6. **So sánh thêm với MLP baseline** từ Lab 1  
7. **Confusion matrix** của mô hình tốt nhất  
8. **Phân tích 3–5 mẫu dự đoán sai**

---

## 14. Tiêu chí chấm gợi ý
- Đúng pipeline dữ liệu và dùng đúng `NEU-CLS.zip`: **15%**  
- CNN from scratch chạy đúng, có validation và early stopping: **20%**  
- Transfer learning chạy đúng, freeze/fine-tune hợp lý: **20%**  
- W&B logging đầy đủ và so sánh rõ ràng: **20%**  
- Bảng so sánh + kết luận có căn cứ: **15%**  
- Phân tích lỗi dự đoán sai và trả lời thảo luận: **10%**

---

## 15. Kết luận học thuật của lab
Lab 2 giúp sinh viên chuyển từ tư duy:
- “biết cách train một mô hình”  

sang:
- “biết chọn kiến trúc phù hợp với bản chất dữ liệu ảnh”,
- “biết khi nào cần CNN từ đầu, khi nào nên tận dụng transfer learning”,
- “biết dùng W&B để đưa ra kết luận dựa trên bằng chứng thực nghiệm”.

Đây là bước nối logic và tự nhiên sau Lab 1. Nếu Lab 1 xây nền cho **training discipline**, thì Lab 2 xây nền cho **modeling choice trên dữ liệu ảnh**.
