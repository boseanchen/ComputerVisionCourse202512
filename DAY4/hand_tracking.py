"""
MediaPipe 手部追蹤範例
最簡單的手部偵測與關鍵點繪製

使用方式:
    python hand_tracking.py

安裝套件:
    pip install mediapipe opencv-python
"""
import numpy as np
import cv2
import mediapipe as mp
import time

# 全域變數來儲存 trackbar 的值
params = {
    'max_num_hands': 2,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'draw_trail': 0,
    'trail_duration': 2,
}

# Trackbar 的回呼函式 (不做任何事，但 createTrackbar 需要它)
def on_change(val):
    pass

# 滑鼠點擊事件的回呼函式
def on_mouse_click(event, x, y, flags, param):
    global running
    # 如果在按鈕範圍內左鍵點擊
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and 10 <= y <= 40:
            running = False # 設定全域變數來停止主迴圈

# 建立 MediaPipe 手部偵測器
def create_hands_detector(p):
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=int(p['max_num_hands']),
        min_detection_confidence=p['min_detection_confidence'],
        min_tracking_confidence=p['min_tracking_confidence']
    )

def main():
    global running # 使用全域變數來控制主迴圈

    # 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # 初始化攝影機
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("正在啟動攝影機與視窗...")

    # ---- 視窗初始化 "預熱" ----
    # 1. 僅建立視窗名稱
    cv2.namedWindow("Hand Tracking")

    # 2. 讀取第一幀畫面來確認攝影機與顯示視窗
    ret, frame = cap.read()
    if not ret:
        print("錯誤: 無法從攝影機讀取畫面。請確認攝影機是否正常連接。")
        cap.release()
        return

    # 3. 立即顯示第一幀畫面並處理GUI事件，確保視窗已建立
    cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
    cv2.waitKey(1)

    # 4. 在確認視窗存在後，才建立 Trackbar 和滑鼠事件
    print("手部追蹤已啟動，可使用滑桿調整參數，或點擊 'Exit' 按鈕退出。")
    cv2.createTrackbar("Max Hands", "Hand Tracking", params['max_num_hands'], 4, on_change)
    cv2.createTrackbar("Detection Conf", "Hand Tracking", int(params['min_detection_confidence'] * 100), 100, on_change)
    cv2.createTrackbar("Tracking Conf", "Hand Tracking", int(params['min_tracking_confidence'] * 100), 100, on_change)
    cv2.createTrackbar("Draw Trail", "Hand Tracking", params['draw_trail'], 1, on_change)
    cv2.createTrackbar("Trail Dura", "Hand Tracking", params['trail_duration'], 10, on_change)
    cv2.setMouseCallback("Hand Tracking", on_mouse_click)

    # ---- 初始化完成，設定偵測器與迴圈變數 ----
    hands = create_hands_detector(params)
    current_params = params.copy()
    trail_points = []
    running = True

    while running and cap.isOpened():
        # 讀取影像
        ret, frame = cap.read()
        if not ret:
            break

        # 水平翻轉 (鏡像) 並轉換顏色
        frame = cv2.flip(frame, 1).copy() # 使用 .copy() 確保影像陣列是可寫入的
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 執行手部偵測
        results = hands.process(rgb_frame)

        # ---- 根據偵測結果與參數進行繪圖 ----
        # 為了安全起見，所有繪圖都在 RGB 影像上完成，最後再轉回 BGR 顯示

        # 確保 rgb_frame 是可寫入的
        rgb_frame.flags.writeable = True

        draw_trail_enabled = params['draw_trail']
        trail_duration = params['trail_duration']

        # 繪製漸變軌跡 (在 rgb_frame 上)
        if draw_trail_enabled and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = rgb_frame.shape
            current_pos = (int(lm8.x * w), int(lm8.y * h))
            trail_points.append({'pos': current_pos, 'time': time.time()})
        
        now = time.time()
        trail_points = [p for p in trail_points if now - p['time'] < trail_duration]

        if draw_trail_enabled and len(trail_points) > 1:
            for i in range(1, len(trail_points)):
                age_ratio = (now - trail_points[i]['time']) / trail_duration
                alpha = min(age_ratio, 1.0)
                # MediaPipe 使用 RGB 顏色 (Red, Green, Blue)
                color = (int((1 - alpha) * 255), 0, int(alpha * 255))
                cv2.line(rgb_frame, trail_points[i-1]['pos'], trail_points[i]['pos'], color, thickness=3)

        # 繪製手部關鍵點 (在 rgb_frame 上)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 將最終繪製好的 RGB 影像轉回 BGR 以便 cv2.imshow 顯示
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # ---- 繪製 UI (在最終的 BGR frame 上) ----
        control_panel = np.zeros((50, frame.shape[1], 3), dtype="uint8")

        # 強制加上偵錯，檢查 control_panel 的維度是否正確
        if len(control_panel.shape) != 3 or control_panel.shape[2] != 3:
            # 如果維度不對，這會引發一個更容易理解的錯誤
            raise RuntimeError(f"Debug Check Failed: control_panel shape is {control_panel.shape}, but expected 3 dimensions with color channel.")

        control_panel[:] = (50, 50, 50)
        cv2.rectangle(control_panel, (10, 10), (110, 40), (0, 0, 200), -1)
        cv2.putText(control_panel, "Exit", (35, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frame[0:50, :] = control_panel

        # 顯示畫面
        cv2.imshow("Hand Tracking", frame)

        # ---- 事件處理與參數更新 ----
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        max_hands = cv2.getTrackbarPos("Max Hands", "Hand Tracking")
        if max_hands == 0:
            max_hands = 1
            cv2.setTrackbarPos("Max Hands", "Hand Tracking", 1)
        
        params['max_num_hands'] = max_hands
        params['min_detection_confidence'] = cv2.getTrackbarPos("Detection Conf", "Hand Tracking") / 100.0
        params['min_tracking_confidence'] = cv2.getTrackbarPos("Tracking Conf", "Hand Tracking") / 100.0
        params['draw_trail'] = cv2.getTrackbarPos("Draw Trail", "Hand Tracking")
        params['trail_duration'] = cv2.getTrackbarPos("Trail Dura", "Hand Tracking")

        if params != current_params:
            print("參數變更，重新初始化偵測器...")
            hands.close()
            hands = create_hands_detector(params)
            current_params = params.copy()

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("程式已結束。")


if __name__ == "__main__":
    main()