# DAY3 - YOLOv11 硬幣檢測

使用 YOLOv11 訓練台幣硬幣檢測模型，可辨識 1、5、10、50 元硬幣的正反面。

## 資料集結構

```
DAY3/
├── data.yaml            # 資料集配置檔
├── train/               # 訓練集 (120 張)
│   ├── images/
│   └── labels/
├── valid/               # 驗證集 (6 張)
│   ├── images/
│   └── labels/
├── test/                # 測試集 (6 張)
│   ├── images/
│   └── labels/
├── train_yolov11.py     # 訓練腳本
├── inference.py         # 命令列推理腳本
├── yolo_gui.py          # GUI 圖形介面應用
├── check_environment.py # 環境檢查工具
└── README.md            # 說明文檔
```

## 類別說明

| 類別 | 說明 |
|------|------|
| 1h / 1t | 1 元硬幣 (正面/反面) |
| 5h / 5t | 5 元硬幣 (正面/反面) |
| 10h / 10t | 10 元硬幣 (正面/反面) |
| 50h / 50t | 50 元硬幣 (正面/反面) |

## 環境安裝

```bash
# 安裝 ultralytics (YOLOv11)
pip install ultralytics

# 安裝 OpenCV (推理需要)
pip install opencv-python

# 安裝 CustomTkinter (GUI 需要)
pip install customtkinter
```

## 環境檢查

在開始訓練前，建議先執行環境檢查腳本:

```bash
python check_environment.py
```

此腳本會檢查:
- Python 版本
- PyTorch 安裝與版本
- CUDA/GPU 可用性
- GPU 記憶體狀態
- Ultralytics (YOLOv11) 安裝
- OpenCV 安裝
- 其他相關套件

並提供 batch size 建議與訓練指令。

## 訓練模型

### 基本訓練

```bash
cd DAY3
python train_yolov11.py
```

### 進階選項

```bash
# 使用較大模型，訓練 200 輪
python train_yolov11.py --model-size m --epochs 200

# 使用 CPU 訓練
python train_yolov11.py --device cpu

# 從中斷處繼續訓練
python train_yolov11.py --resume
```

### 模型大小選擇

| 大小 | 參數 | 速度 | 準確度 | 適用場景 |
|------|------|------|--------|----------|
| n (nano) | 最少 | 最快 | 較低 | 即時應用、邊緣裝置 |
| s (small) | 少 | 快 | 中等 | 平衡效能 |
| m (medium) | 中等 | 中等 | 較高 | 一般應用 |
| l (large) | 多 | 較慢 | 高 | 高準確度需求 |
| x (xlarge) | 最多 | 最慢 | 最高 | 研究、高精度需求 |

## 使用模型

### 即時攝影機偵測

```bash
python inference.py --model runs/detect/coin_detector/weights/best.pt --source 0
```

按鍵操作:
- `q`: 退出
- `s`: 截圖

### 單張圖片預測

```bash
python inference.py --model runs/detect/coin_detector/weights/best.pt --source test_image.jpg
```

### 影片檔案預測

```bash
python inference.py --model runs/detect/coin_detector/weights/best.pt --source video.mp4
```

### 調整信心閾值

```bash
# 提高信心閾值 (減少誤判)
python inference.py --model best.pt --source 0 --conf 0.5
```

## GUI 圖形介面

提供現代化的圖形介面，方便操作:

```bash
python yolo_gui.py
```

### 功能特色:
- 選擇並載入 .pt 模型檔案
- 支援圖片、影片、攝影機三種輸入來源
- 即時調整信心閾值
- 自動計算偵測到的硬幣總金額
- 不同面額硬幣以不同顏色標示
- 深色主題現代化介面

### 使用步驟:
1. 點擊「選擇模型檔案」載入訓練好的 .pt 模型
2. 選擇輸入來源:
   - 📷 選擇圖片: 偵測單張圖片
   - 🎬 選擇影片: 偵測影片檔案
   - 📹 開啟攝影機: 即時攝影機偵測
3. 調整信心閾值滑桿以優化偵測結果
4. 查看右側面板的偵測結果與總金額

## 驗證模型

```bash
python train_yolov11.py --mode val --model-path runs/detect/coin_detector/weights/best.pt
```

## 匯出模型

```bash
# 匯出為 ONNX 格式
python train_yolov11.py --mode export --model-path best.pt --export-format onnx

# 匯出為 TensorRT 格式 (需要 TensorRT 環境)
python train_yolov11.py --mode export --model-path best.pt --export-format engine
```

## 訓練輸出

訓練完成後，結果會儲存在 `runs/detect/coin_detector/`:

```
runs/detect/coin_detector/
├── weights/
│   ├── best.pt    # 最佳模型
│   └── last.pt    # 最終模型
├── results.png    # 訓練曲線圖
├── confusion_matrix.png
├── PR_curve.png
└── ...
```

## 資料來源

資料集來自 Roboflow:
- 專案: [ProjectCOIN](https://universe.roboflow.com/aicourse-dvibo/projectcoin/dataset/1)
- 授權: CC BY 4.0

## 常見問題

### Q: GPU 記憶體不足怎麼辦?

降低 batch size:
```bash
python train_yolov11.py --batch-size 8
```

### Q: 訓練中斷了怎麼辦?

使用 `--resume` 繼續訓練:
```bash
python train_yolov11.py --resume
```

### Q: 如何提高準確度?

1. 增加訓練資料
2. 使用較大的模型 (m/l/x)
3. 增加訓練輪數
4. 調整資料增強參數
