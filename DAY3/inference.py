"""
YOLOv11 硬幣檢測即時推理腳本
支援攝影機即時偵測與單張圖片預測

作者: AI Course
日期: 2024
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse


# 硬幣面額對應表
COIN_VALUES = {
    '1h': 1, '1t': 1,
    '5h': 5, '5t': 5,
    '10h': 10, '10t': 10,
    '50h': 50, '50t': 50,
    '0': 0,
    'test': 0,
}


def load_model(model_path: str):
    """載入 YOLOv11 模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型: {model_path}")
    return YOLO(model_path)


def calculate_total(detections: list) -> int:
    """計算偵測到的硬幣總金額"""
    total = 0
    for class_name in detections:
        total += COIN_VALUES.get(class_name, 0)
    return total


def process_results(results, frame, show_total: bool = True):
    """處理偵測結果並繪製在畫面上"""

    detected_coins = []
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            # 取得邊界框座標
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = names[cls_id]

            detected_coins.append(class_name)

            # 繪製邊界框
            color = (0, 255, 0)  # 綠色
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # 繪製標籤
            label = f"{class_name} {conf:.2f}"
            value = COIN_VALUES.get(class_name, 0)
            if value > 0:
                label += f" (${value})"

            # 計算標籤背景大小
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color, -1
            )
            cv2.putText(
                annotated_frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

    # 顯示總金額
    if show_total and detected_coins:
        total = calculate_total(detected_coins)
        total_text = f"Total: ${total}"
        cv2.putText(
            annotated_frame, total_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )

    return annotated_frame, detected_coins


def run_webcam(model_path: str, camera_id: int = 0, conf: float = 0.25):
    """使用攝影機進行即時偵測"""

    print(f"載入模型: {model_path}")
    model = load_model(model_path)

    print(f"開啟攝影機 {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("錯誤: 無法開啟攝影機")
        return

    # 設定攝影機解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("按 'q' 退出, 's' 截圖")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("錯誤: 無法讀取畫面")
            break

        # 水平翻轉 (鏡像)
        frame = cv2.flip(frame, 1)

        # 執行偵測
        results = model.predict(frame, conf=conf, verbose=False)

        # 處理並繪製結果
        annotated_frame, coins = process_results(results, frame)

        # 顯示畫面
        cv2.imshow("YOLOv11 Coin Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 截圖
            filename = f"screenshot_{len(os.listdir('.'))}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"截圖已儲存: {filename}")

    cap.release()
    cv2.destroyAllWindows()


def predict_image(model_path: str, image_path: str, conf: float = 0.25, save: bool = True):
    """對單張圖片進行預測"""

    print(f"載入模型: {model_path}")
    model = load_model(model_path)

    print(f"讀取圖片: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("錯誤: 無法讀取圖片")
        return

    # 執行偵測
    results = model.predict(frame, conf=conf, verbose=False)

    # 處理並繪製結果
    annotated_frame, coins = process_results(results, frame)

    # 計算總金額
    total = calculate_total(coins)

    print("\n偵測結果:")
    print(f"  偵測到的硬幣: {coins}")
    print(f"  總金額: ${total}")

    if save:
        output_path = image_path.rsplit('.', 1)[0] + "_result.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"  結果已儲存: {output_path}")

    # 顯示結果
    cv2.imshow("Detection Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return coins, total


def predict_video(model_path: str, video_path: str, conf: float = 0.25, save: bool = True):
    """對影片進行預測"""

    print(f"載入模型: {model_path}")
    model = load_model(model_path)

    print(f"讀取影片: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("錯誤: 無法讀取影片")
        return

    # 取得影片資訊
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 設定輸出影片
    if save:
        output_path = video_path.rsplit('.', 1)[0] + "_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("處理中... 按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 執行偵測
        results = model.predict(frame, conf=conf, verbose=False)

        # 處理並繪製結果
        annotated_frame, coins = process_results(results, frame)

        if save:
            out.write(annotated_frame)

        # 顯示畫面
        cv2.imshow("Video Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save:
        out.release()
        print(f"結果已儲存: {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 硬幣檢測推理腳本")

    parser.add_argument("--model", type=str, required=True,
                       help="模型路徑 (.pt 檔案)")
    parser.add_argument("--source", type=str, default="0",
                       help="來源: 0=攝影機, 或圖片/影片路徑")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="信心閾值")
    parser.add_argument("--no-save", action="store_true",
                       help="不儲存結果")

    args = parser.parse_args()

    # 判斷來源類型
    if args.source == "0" or args.source.isdigit():
        # 攝影機
        run_webcam(args.model, int(args.source), args.conf)
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # 影片
        predict_video(args.model, args.source, args.conf, not args.no_save)
    else:
        # 圖片
        predict_image(args.model, args.source, args.conf, not args.no_save)
