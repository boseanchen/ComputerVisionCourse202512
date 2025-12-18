"""
MediaPipe 手部追蹤 + MNIST 手寫數字辨識
在辨識區域內用食指畫數字，自動辨識並顯示結果

使用方式:
    python hand_digit_recognition.py

流程:
    1. 食指進入辨識區域
    2. 0.5 秒後開始記錄軌跡 (框變綠色)
    3. 0.8 秒後停止記錄 (框變回白色)
    4. 將軌跡送入 MNIST 模型辨識
    5. 結果顯示在左上角
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import os
import sys

# 加入 DAY2 路徑以載入模型
DAY2_PATH = os.path.join(os.path.dirname(__file__), "..", "DAY2", "01_MNIST")
sys.path.append(DAY2_PATH)


# ============== MNIST CNN 模型定義 (與 DAY2 相同) ==============
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============== 設定 ==============
MODEL_PATH = r"D:\AWORKSPACE\Github\ComputerVisionCourse202512\DAY2\models\mnist_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 辨識區域設定
ZONE_WIDTH = 300
ZONE_HEIGHT = 300

# 時間設定
ENTER_DELAY = 1      # 進入區域後等待秒數
RECORD_DURATION = 1.8  # 記錄持續秒數

# 顏色設定 (BGR)
COLOR_IDLE = (255, 255, 255)       # 白色 - 等待中
COLOR_WAITING = (0, 255, 255)     # 黃色 - 已進入，等待開始
COLOR_RECORDING = (0, 255, 0)     # 綠色 - 記錄中
COLOR_TRAIL = (0, 200, 255)       # 橙色 - 軌跡顏色


# ============== 載入模型 ==============
def load_model():
    """載入 MNIST CNN 模型"""
    model = SimpleCNN().to(DEVICE)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"模型載入成功: {MODEL_PATH}")
    else:
        print(f"警告: 找不到模型 {MODEL_PATH}")
        print("請先執行 DAY2/01_MNIST/train.py 訓練模型")
        return None

    return model


# ============== 軌跡轉換為 MNIST 格式 ==============
def trail_to_mnist_image(trail_points, zone_rect):
    """
    將軌跡點轉換為 28x28 的 MNIST 格式圖像

    Args:
        trail_points: 軌跡點列表 [(x, y), ...]
        zone_rect: 辨識區域 (x, y, w, h)

    Returns:
        28x28 的灰階圖像 (numpy array)
    """
    if len(trail_points) < 2:
        return None

    zx, zy, zw, zh = zone_rect

    # 建立空白畫布 (白底黑字，之後反轉)
    canvas_size = 280  # 先在較大的畫布上繪製
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # 將軌跡點轉換到畫布座標
    points = []
    for px, py in trail_points:
        # 相對於 zone 的座標
        rx = px - zx
        ry = py - zy
        # 縮放到畫布大小
        cx = int(rx / zw * canvas_size)
        cy = int(ry / zh * canvas_size)
        # 確保在範圍內
        cx = max(0, min(canvas_size - 1, cx))
        cy = max(0, min(canvas_size - 1, cy))
        points.append((cx, cy))

    # 繪製軌跡
    for i in range(1, len(points)):
        cv2.line(canvas, points[i-1], points[i], 255, thickness=20)

    # 找到數字的邊界框
    coords = np.column_stack(np.where(canvas > 0))
    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 裁切數字區域
    digit = canvas[y_min:y_max+1, x_min:x_max+1]

    # 加入邊距
    pad = 20
    digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # 縮放到 28x28
    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    # 正規化
    digit = digit.astype(np.float32) / 255.0

    return digit


# ============== 預測數字 ==============
def predict_digit(model, image):
    """
    使用模型預測數字

    Args:
        model: MNIST CNN 模型
        image: 28x28 灰階圖像

    Returns:
        預測的數字 (0-9)
    """
    if model is None or image is None:
        return None

    # 正規化 (MNIST 的均值和標準差)
    image = (image - 0.1307) / 0.3081

    # 轉換為 tensor
    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # 加入 batch 和 channel 維度
    tensor = tensor.to(DEVICE)

    # 預測
    with torch.no_grad():
        output = model(tensor)
        _, predicted = output.max(1)

    return predicted.item()


# ============== 主程式 ==============
def main():
    # 載入模型
    print("載入 MNIST 模型...")
    model = load_model()

    # 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # 只追蹤一隻手
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("無法開啟攝影機")
        return

    frame_h, frame_w = frame.shape[:2]

    # 計算辨識區域位置 (畫面中間)
    zone_x = (frame_w - ZONE_WIDTH) // 2
    zone_y = (frame_h - ZONE_HEIGHT) // 2
    zone_rect = (zone_x, zone_y, ZONE_WIDTH, ZONE_HEIGHT)

    print(f"畫面大小: {frame_w}x{frame_h}")
    print(f"辨識區域: ({zone_x}, {zone_y}) - {ZONE_WIDTH}x{ZONE_HEIGHT}")
    print()
    print("操作說明:")
    print("  1. 將食指移入中間的辨識區域")
    print("  2. 等待 0.5 秒 (框變黃色)")
    print("  3. 框變綠色時開始畫數字 (0.8 秒)")
    print("  4. 結果顯示在左上角")
    print("  按 'c' 清除結果，按 'q' 退出")

    # 狀態變數
    state = "idle"  # idle, waiting, recording
    enter_time = None
    record_start_time = None
    trail_points = []
    recognized_digits = []  # 辨識結果列表

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()

        # 轉換顏色
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 取得食指位置
        finger_in_zone = False
        finger_pos = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 繪製手部骨架
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # 取得食指尖位置
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                fx = int(index_tip.x * frame_w)
                fy = int(index_tip.y * frame_h)
                finger_pos = (fx, fy)

                # 檢查是否在辨識區域內
                if (zone_x <= fx <= zone_x + ZONE_WIDTH and
                    zone_y <= fy <= zone_y + ZONE_HEIGHT):
                    finger_in_zone = True

                # 在食指尖畫圓
                cv2.circle(frame, (fx, fy), 10, (0, 255, 0), -1)

        # 狀態機
        if state == "idle":
            if finger_in_zone:
                state = "waiting"
                enter_time = current_time
                trail_points = []
                print("食指進入區域，等待開始...")

        elif state == "waiting":
            if not finger_in_zone:
                state = "idle"
                enter_time = None
                print("食指離開區域，重置")
            elif current_time - enter_time >= ENTER_DELAY:
                state = "recording"
                record_start_time = current_time
                trail_points = []
                print("開始記錄軌跡!")

        elif state == "recording":
            # 記錄軌跡
            if finger_pos and finger_in_zone:
                trail_points.append(finger_pos)

            # 檢查是否超時
            if current_time - record_start_time >= RECORD_DURATION:
                print(f"記錄完成，共 {len(trail_points)} 個點")

                # 轉換軌跡並辨識
                if len(trail_points) > 10:
                    digit_image = trail_to_mnist_image(trail_points, zone_rect)
                    if digit_image is not None:
                        result = predict_digit(model, digit_image)
                        if result is not None:
                            recognized_digits.append(result)
                            print(f"辨識結果: {result}")

                            # 最多保留 10 個結果
                            if len(recognized_digits) > 10:
                                recognized_digits.pop(0)
                else:
                    print("軌跡點太少，跳過辨識")

                # 重置狀態
                state = "idle"
                enter_time = None
                record_start_time = None
                trail_points = []

        # 決定框的顏色
        if state == "idle":
            zone_color = COLOR_IDLE
        elif state == "waiting":
            zone_color = COLOR_WAITING
        else:  # recording
            zone_color = COLOR_RECORDING

        # 繪製辨識區域
        cv2.rectangle(
            frame,
            (zone_x, zone_y),
            (zone_x + ZONE_WIDTH, zone_y + ZONE_HEIGHT),
            zone_color, 3
        )

        # 繪製軌跡
        if len(trail_points) > 1:
            for i in range(1, len(trail_points)):
                cv2.line(
                    frame,
                    trail_points[i-1],
                    trail_points[i],
                    COLOR_TRAIL, 4
                )

        # 顯示狀態文字
        if state == "idle":
            status_text = "Move finger into zone"
        elif state == "waiting":
            remaining = max(0, ENTER_DELAY - (current_time - enter_time))
            status_text = f"Wait... {remaining:.1f}s"
        else:  # recording
            remaining = max(0, RECORD_DURATION - (current_time - record_start_time))
            status_text = f"Drawing! {remaining:.1f}s"

        cv2.putText(
            frame,
            status_text,
            (zone_x, zone_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2
        )

        # 顯示辨識結果 (左上角，由左至右排列)
        if recognized_digits:
            result_text = "Results: " + " ".join(map(str, recognized_digits))
            cv2.putText(
                frame,
                result_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )

            # 繪製數字方框
            for i, digit in enumerate(recognized_digits):
                box_x = 20 + i * 50
                box_y = 60
                cv2.rectangle(frame, (box_x, box_y), (box_x + 45, box_y + 60), (50, 50, 50), -1)
                cv2.rectangle(frame, (box_x, box_y), (box_x + 45, box_y + 60), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    str(digit),
                    (box_x + 10, box_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
                )

        # 顯示操作提示
        cv2.putText(
            frame,
            "Press 'c' to clear, 'q' to quit",
            (20, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        # 顯示畫面
        cv2.imshow("Hand Digit Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            recognized_digits = []
            print("清除結果")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
