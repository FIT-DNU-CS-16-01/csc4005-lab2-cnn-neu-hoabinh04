# W&B Guide – CSC4005 Lab 2

## 1. Tạo tài khoản và login
```bash
wandb login
```
Dán API key khi được yêu cầu.

## 2. Project thống nhất
Dùng project name:
```text
csc4005-lab2-neu-cnn
```

## 3. Vì sao phải dùng W&B trong Lab 2?
Vì Lab 2 yêu cầu so sánh:
- MLP baseline từ Lab 1
- CNN from scratch
- transfer learning

W&B giúp lưu:
- config của từng run
- learning curves
- best validation metrics
- ảnh confusion matrix
- thời gian train/epoch

## 4. Những trường nên log
Tối thiểu:
- model_name
- train_mode
- optimizer
- lr
- weight_decay
- dropout
- img_size
- batch_size
- train_loss
- val_loss
- train_acc
- val_acc
- epoch_time_sec

## 5. So sánh trong dashboard
Sau khi có >= 3 runs, vào project và so sánh:
- best_val_acc
- test_acc
- avg_epoch_time_sec
- trainable_params
- curves
