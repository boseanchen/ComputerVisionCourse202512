"""
YOLOv11 硬幣檢測模型訓練腳本
Project COIN - 台幣硬幣檢測與辨識

作者: AI Course
日期: 2024
"""

from ultralytics import YOLO
import os
import argparse


def train_model(
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "coin_detector",
    resume: bool = False,
):
    """
    訓練 YOLOv11 硬幣檢測模型

    參數:
        model_size: 模型大小 (n/s/m/l/x)
                   n: nano (最小最快)
                   s: small
                   m: medium
                   l: large
                   x: xlarge (最大最準)
        epochs: 訓練輪數
        batch_size: 批次大小
        img_size: 輸入圖片大小
        device: 使用的設備 (0=GPU, cpu=CPU)
        project: 輸出專案目錄
        name: 實驗名稱
        resume: 是否從上次中斷處繼續訓練
    """

    # 取得腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, "data.yaml")

    # 選擇預訓練模型
    model_name = f"yolo11{model_size}.pt"

    print("=" * 60)
    print("YOLOv11 硬幣檢測模型訓練")
    print("=" * 60)
    print(f"模型大小: {model_size} ({model_name})")
    print(f"訓練輪數: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"圖片大小: {img_size}")
    print(f"設備: {device}")
    print(f"資料配置: {data_yaml}")
    print("=" * 60)

    # 載入模型
    if resume:
        # 從上次訓練繼續
        last_weights = os.path.join(project, name, "weights", "last.pt")
        if os.path.exists(last_weights):
            model = YOLO(last_weights)
            print(f"從 {last_weights} 繼續訓練")
        else:
            print(f"找不到 {last_weights}，使用預訓練模型開始")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)

    # 開始訓練
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        # 資料增強設定
        hsv_h=0.015,      # 色調變化
        hsv_s=0.7,        # 飽和度變化
        hsv_v=0.4,        # 明度變化
        degrees=10,       # 旋轉角度
        translate=0.1,    # 平移
        scale=0.5,        # 縮放
        flipud=0.0,       # 上下翻轉 (硬幣不需要)
        fliplr=0.5,       # 左右翻轉
        mosaic=1.0,       # Mosaic 資料增強
        mixup=0.0,        # MixUp 資料增強
        # 訓練設定
        patience=50,      # 早停耐心值
        save=True,        # 儲存模型
        save_period=10,   # 每 N 輪儲存一次
        val=True,         # 訓練時驗證
        plots=True,       # 生成訓練圖表
        verbose=True,     # 詳細輸出
    )

    print("\n" + "=" * 60)
    print("訓練完成!")
    print("=" * 60)
    print(f"最佳模型: {project}/{name}/weights/best.pt")
    print(f"最終模型: {project}/{name}/weights/last.pt")

    return results


def validate_model(model_path: str, data_yaml: str = None):
    """驗證模型效能"""

    if data_yaml is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_yaml = os.path.join(script_dir, "data.yaml")

    model = YOLO(model_path)
    results = model.val(data=data_yaml)

    print("\n驗證結果:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    return results


def predict_image(model_path: str, source: str, conf: float = 0.25, save: bool = True):
    """使用模型進行預測"""

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=False,
    )

    return results


def export_model(model_path: str, format: str = "onnx"):
    """
    匯出模型到其他格式

    支援格式: onnx, torchscript, openvino, engine (TensorRT), coreml
    """

    model = YOLO(model_path)
    model.export(format=format)
    print(f"模型已匯出為 {format} 格式")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 硬幣檢測訓練腳本")

    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "val", "predict", "export"],
                       help="執行模式: train/val/predict/export")
    parser.add_argument("--model-size", type=str, default="n",
                       choices=["n", "s", "m", "l", "x"],
                       help="模型大小")
    parser.add_argument("--epochs", type=int, default=100,
                       help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--img-size", type=int, default=640,
                       help="圖片大小")
    parser.add_argument("--device", type=str, default="0",
                       help="設備 (0=GPU, cpu=CPU)")
    parser.add_argument("--resume", action="store_true",
                       help="從上次中斷處繼續訓練")
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型路徑 (用於 val/predict/export)")
    parser.add_argument("--source", type=str, default=None,
                       help="預測來源 (圖片/影片路徑或 0 表示攝影機)")
    parser.add_argument("--export-format", type=str, default="onnx",
                       help="匯出格式")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            resume=args.resume,
        )

    elif args.mode == "val":
        if args.model_path is None:
            print("錯誤: 驗證模式需要指定 --model-path")
        else:
            validate_model(args.model_path)

    elif args.mode == "predict":
        if args.model_path is None or args.source is None:
            print("錯誤: 預測模式需要指定 --model-path 和 --source")
        else:
            predict_image(args.model_path, args.source)

    elif args.mode == "export":
        if args.model_path is None:
            print("錯誤: 匯出模式需要指定 --model-path")
        else:
            export_model(args.model_path, args.export_format)
