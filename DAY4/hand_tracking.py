"""
MediaPipe 手部追蹤範例
食指軌跡效果 - 維持 2 秒後漸變淡出

使用方式:
    python hand_tracking.py

安裝套件:
    pip install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque


# 軌跡設定
TRAIL_DURATION = 2.0  # 軌跡維持秒數
TRAIL_COLOR = (0, 255, 255)  # 軌跡顏色 (BGR: 黃色)
TRAIL_THICKNESS = 4  # 軌跡粗細


def main():
    # 初始化 MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # 建立手部偵測器
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=20,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("手部追蹤已啟動，按 'q' 退出")
    print("用食指畫出軌跡，軌跡會在 2 秒後漸變淡出")

    # 儲存軌跡點 (x, y, timestamp)
    trail_points = deque()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 水平翻轉 (鏡像)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()

        # BGR 轉 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 執行手部偵測
        results = hands.process(rgb_frame)

        # 偵測到手時，記錄食指位置
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 繪製手部骨架
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # 取得食指尖 (8號點) 座標
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx = int(index_tip.x * w)
                cy = int(index_tip.y * h)

                # 加入軌跡點
                trail_points.append((cx, cy, current_time))

                # 在食指尖畫一個圓
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)

        # 移除過期的軌跡點 (超過 TRAIL_DURATION 秒)
        while trail_points and (current_time - trail_points[0][2]) > TRAIL_DURATION:
            trail_points.popleft()

        # 繪製漸變淡出的軌跡
        if len(trail_points) > 1:
            # 建立透明圖層
            overlay = frame.copy()

            points_list = list(trail_points)
            for i in range(1, len(points_list)):
                # 計算這個點的存活時間比例 (0 = 剛畫, 1 = 快消失)
                age = current_time - points_list[i][2]
                age_ratio = age / TRAIL_DURATION  # 0.0 ~ 1.0

                # 計算透明度 (越舊越透明)
                alpha = 1.0 - age_ratio  # 1.0 ~ 0.0

                # 計算顏色強度 (越舊越暗)
                color_intensity = alpha
                color = (
                    int(TRAIL_COLOR[0] * color_intensity),
                    int(TRAIL_COLOR[1] * color_intensity),
                    int(TRAIL_COLOR[2] * color_intensity)
                )

                # 計算線條粗細 (越舊越細)
                thickness = max(1, int(TRAIL_THICKNESS * alpha))

                # 取得前後兩點
                pt1 = (points_list[i - 1][0], points_list[i - 1][1])
                pt2 = (points_list[i][0], points_list[i][1])

                # 畫線
                cv2.line(overlay, pt1, pt2, color, thickness)

            # 混合圖層
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # 顯示提示文字
        cv2.putText(
            frame,
            f"Trail points: {len(trail_points)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # 顯示畫面
        cv2.imshow("Hand Tracking - Finger Trail", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
