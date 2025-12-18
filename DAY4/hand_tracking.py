"""
MediaPipe 手部追蹤範例
一個經過重構和優化的版本，將程式碼封裝在一個類別中。

使用方式:
    python hand_tracking.py

安裝套件:
    pip install mediapipe opencv-python
"""
import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque

class HandTrackerApp:
    """
    將手部追蹤應用程式封裝在一個類別中，以獲得更好的結構和狀態管理。
    """
    def __init__(self):
        # --- 常數與設定 ---
        self.WINDOW_NAME = "Hand Tracking"
        self.WIDTH = 1280
        self.HEIGHT = 720

        # --- 狀態變數 ---
        self.running = True
        self.hands_detector = None
        # 使用 collections.deque 來高效地儲存固定長度的軌跡點
        self.trail_points = deque()

        # --- 參數字典 ---
        self.params = {
            'max_num_hands': 2,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'draw_trail': 0,
            'trail_duration': 2,
        }
        self.current_params = self.params.copy()

        # --- MediaPipe 初始化 ---
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

    def _create_hands_detector(self):
        """根據目前的參數建立一個新的 MediaPipe Hands 偵測器實例。"""
        print("正在初始化 MediaPipe Hands 偵測器...")
        return self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(self.params['max_num_hands']),
            min_detection_confidence=self.params['min_detection_confidence'],
            min_tracking_confidence=self.params['min_tracking_confidence']
        )

    def _setup_gui(self, cap):
        """設定 OpenCV 視窗、滑桿和滑鼠回呼。"""
        cv2.namedWindow(self.WINDOW_NAME)

        # 透過顯示第一幀來"預熱"視窗，確保GUI已準備就緒
        ret, frame = cap.read()
        if not ret:
            print("錯誤：無法從攝影機讀取影像。")
            self.running = False
            return
        cv2.imshow(self.WINDOW_NAME, cv2.flip(frame, 1))
        cv2.waitKey(1)

        # 建立滑桿
        cv2.createTrackbar("Max Hands", self.WINDOW_NAME, self.params['max_num_hands'], 4, self._on_change)
        cv2.createTrackbar("Detection Conf", self.WINDOW_NAME, int(self.params['min_detection_confidence'] * 100), 100, self._on_change)
        cv2.createTrackbar("Tracking Conf", self.WINDOW_NAME, int(self.params['min_tracking_confidence'] * 100), 100, self._on_change)
        cv2.createTrackbar("Draw Trail", self.WINDOW_NAME, self.params['draw_trail'], 1, self._on_change)
        cv2.createTrackbar("Trail Dura (s)", self.WINDOW_NAME, self.params['trail_duration'], 10, self._on_change)
        
        # 設定滑鼠回呼
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse_click)
        print("UI 設定完成，正在啟動主迴圈。")

    def _on_change(self, val):
        """滑桿的虛設回呼函式。"""
        pass

    def _on_mouse_click(self, event, x, y, flags, param):
        """處理UI事件的滑鼠點擊回呼函式。"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 離開按鈕的範圍
            if 10 <= x <= 110 and 10 <= y <= 40:
                self.running = False

    def _update_params_from_trackbars(self):
        """從滑桿讀取數值並更新參數字典。"""
        max_hands = cv2.getTrackbarPos("Max Hands", self.WINDOW_NAME)
        # 確保至少偵測1隻手
        if max_hands == 0:
            max_hands = 1
            cv2.setTrackbarPos("Max Hands", self.WINDOW_NAME, 1)

        self.params['max_num_hands'] = max_hands
        self.params['min_detection_confidence'] = cv2.getTrackbarPos("Detection Conf", self.WINDOW_NAME) / 100.0
        self.params['min_tracking_confidence'] = cv2.getTrackbarPos("Tracking Conf", self.WINDOW_NAME) / 100.0
        self.params['draw_trail'] = cv2.getTrackbarPos("Draw Trail", self.WINDOW_NAME)
        # 確保軌跡持續時間至少為1秒
        trail_duration = cv2.getTrackbarPos("Trail Dura (s)", self.WINDOW_NAME)
        self.params['trail_duration'] = max(1, trail_duration)

        # 如果參數已變更，則重新建立偵測器
        if self.params != self.current_params:
            print("參數已變更，正在重新初始化偵測器...")
            if self.hands_detector:
                self.hands_detector.close()
            self.hands_detector = self._create_hands_detector()
            self.current_params = self.params.copy()
            
    def _draw_overlays(self, frame, results):
        """在影像幀上繪製所有覆蓋圖層：軌跡、標記點和UI。"""
        # 1. 繪製軌跡
        trail_duration = self.params['trail_duration']
        now = time.time()
        
        # 移除舊的軌跡點
        while self.trail_points and now - self.trail_points[0]['time'] > trail_duration:
            self.trail_points.popleft()

        # 如果啟用，新增新的軌跡點
        if self.params['draw_trail'] and results.multi_hand_landmarks:
            # 以第一隻手的食指指尖作為軌跡來源
            hand_landmarks = results.multi_hand_landmarks[0]
            lm8 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            current_pos = (int(lm8.x * w), int(lm8.y * h))
            self.trail_points.append({'pos': current_pos, 'time': now})

        # 繪製軌跡線
        if self.params['draw_trail'] and len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                p1 = self.trail_points[i-1]
                p2 = self.trail_points[i]
                # 根據點的"年齡"計算顏色/透明度
                age_ratio = (now - p2['time']) / trail_duration
                alpha = min(age_ratio, 1.0)
                # BGR 顏色：從紅色(新)漸變到藍色(舊)
                color = (int(alpha * 255), 0, int((1 - alpha) * 255))
                cv2.line(frame, p1['pos'], p2['pos'], color, thickness=3)

        # 2. 繪製手部標記點
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
        # 3. 繪製UI面板
        # 使用 cv2.rectangle 比建立新的 numpy 陣列更有效率
        cv2.rectangle(frame, (0, 0), (self.WIDTH, 50), (50, 50, 50), -1)
        # 離開按鈕
        cv2.rectangle(frame, (10, 10), (110, 40), (0, 0, 200), -1)
        cv2.putText(frame, "Exit", (35, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame

    def run(self):
        """主應用程式迴圈。"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        
        if not cap.isOpened():
            print("錯誤：無法開啟攝影機。")
            return

        # 設定視窗和滑桿
        self._setup_gui(cap)

        # 第一次初始化偵測器
        self.hands_detector = self._create_hands_detector()

        while self.running and cap.isOpened():
            # 1. 讀取影像
            ret, frame = cap.read()
            if not ret:
                print("讀取影像串流結束。")
                break

            # 2. 處理影像
            # 使用 .copy() 來避免因翻轉產生的唯讀問題，並保留原始 BGR 格式用於繪圖
            bgr_frame = cv2.flip(frame, 1).copy()
            # 將影像轉為 RGB 以供 MediaPipe 處理
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            
            # 為了提升效能，可選擇性地將影像標記為不可寫入，以引用方式傳遞
            rgb_frame.flags.writeable = False
            results = self.hands_detector.process(rgb_frame)
            
            # 3. 繪製圖層
            # 我們直接在 BGR 影像上繪圖
            final_frame = self._draw_overlays(bgr_frame, results)

            # 4. 顯示影像
            cv2.imshow(self.WINDOW_NAME, final_frame)

            # 5. 處理事件與更新參數
            # 等待按鍵事件，如果按下 'q' 則停止迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
            
            self._update_params_from_trackbars()

        # --- 清理資源 ---
        print("正在離開應用程式...")
        if self.hands_detector:
            self.hands_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("清理完成。")

if __name__ == "__main__":
    app = HandTrackerApp()
    app.run()
