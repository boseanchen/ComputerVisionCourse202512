"""
MediaPipe 手部追蹤範例
一個經過重構和優化的版本，將程式碼封裝在一個類別中。
新增功能：可透過獨立的 Tkinter 視窗中的按鈕來離開應用程式。

使用方式:
    python hand_tracking.py

安裝套件:
    pip install mediapipe opencv-python Pillow # Pillow for ImageTk (if needed, but ttk should be fine)
    pip install "pyinstaller[optional]" # For win32console and win32gui on Windows
"""
import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque
import tkinter as tk
from tkinter import ttk # For themed widgets (nicer buttons)

# --- Tkinter Exit Button Application ---
class ExitButtonApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("控制面板")
        self.root.geometry("150x60+10+10") # 小視窗，定位在左上角
        self.root.resizable(False, False)

        self._should_exit = False

        # 建立一個風格，讓按鈕看起來更美觀
        style = ttk.Style()
        style.theme_use('clam') # 'clam' 是一個比較現代的佈景主題
        style.configure('TButton', font=('Helvetica', 10, 'bold'),
                        foreground='white', background='#dc3545', # Bootstrap danger red
                        padding=6, relief='raised')
        style.map('TButton',
                  background=[('active', '#c82333'), ('pressed', '#bd2130')],
                  relief=[('pressed', 'sunken')])

        # 建立離開按鈕
        self.exit_button = ttk.Button(self.root, text="離開應用程式", command=self._on_exit_click, style='TButton')
        self.exit_button.pack(pady=5) # 增加一些內邊距

        # 讓視窗始終保持在最上層 (可選，但對控制面板很有用)
        self.root.attributes('-topmost', True)
        
        # 隱藏執行檔的控制台視窗 (僅限Windows)
        try:
            import win32console, win32gui, win32con
            hwnd = win32console.GetConsoleWindow()
            if hwnd:
                win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
        except ImportError:
            pass # 不在Windows上，或win32con未安裝

    def _on_exit_click(self):
        self._should_exit = True

    def get_should_exit(self):
        return self._should_exit

    def update_tkinter(self):
        """處理所有待處理的Tkinter事件。"""
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            # 如果窗口被關閉，Tkinter 會引發 TclError
            self._should_exit = True

    def destroy_tkinter(self):
        """銷毀Tkinter視窗。"""
        if self.root.winfo_exists():
            self.root.destroy()

# --- Hand Tracking Application Class ---
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
        self.trail_points = deque()

        # --- 參數字典 ---
        self.params = {
            'max_num_hands': 2,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'draw_trail': 0,
            'trail_duration': 2,
            'pinch_threshold': 50, # 捏合靈敏度的預設值 (0-100)
        }
        self.current_params = self.params.copy()

        # --- MediaPipe 初始化 ---
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # --- Tkinter 離開按鈕應用程式 ---
        self.exit_app_tk = ExitButtonApp()

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
        """設定 OpenCV 視窗和滑桿。"""
        cv2.namedWindow(self.WINDOW_NAME)

        ret, frame = cap.read()
        if not ret:
            print("錯誤：無法從攝影機讀取影像。")
            self.running = False
            return
        cv2.imshow(self.WINDOW_NAME, cv2.flip(frame, 1))
        cv2.waitKey(1)

        cv2.createTrackbar("Max Hands", self.WINDOW_NAME, self.params['max_num_hands'], 4, self._on_change)
        cv2.createTrackbar("Detection Conf", self.WINDOW_NAME, int(self.params['min_detection_confidence'] * 100), 100, self._on_change)
        cv2.createTrackbar("Tracking Conf", self.WINDOW_NAME, int(self.params['min_tracking_confidence'] * 100), 100, self._on_change)
        cv2.createTrackbar("Draw Trail", self.WINDOW_NAME, self.params['draw_trail'], 1, self._on_change)
        cv2.createTrackbar("Trail Dura (s)", self.WINDOW_NAME, self.params['trail_duration'], 10, self._on_change)
        cv2.createTrackbar("Pinch Thresh", self.WINDOW_NAME, self.params['pinch_threshold'], 100, self._on_change)
        
        # 移除 OpenCV 滑鼠回呼，因為現在使用 Tkinter 按鈕離開
        # cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse_click)
        print("UI 設定完成，正在啟動主迴圈。")

    def _on_change(self, val):
        """滑桿的虛設回呼函式。"""
        pass

    # 移除 _on_mouse_click 方法，因為不再使用 OpenCV 的按鈕互動

    def _update_params_from_trackbars(self):
        """從滑桿讀取數值並更新參數字典。"""
        max_hands = cv2.getTrackbarPos("Max Hands", self.WINDOW_NAME)
        if max_hands == 0:
            max_hands = 1
            cv2.setTrackbarPos("Max Hands", self.WINDOW_NAME, 1)

        self.params['max_num_hands'] = max_hands
        self.params['min_detection_confidence'] = cv2.getTrackbarPos("Detection Conf", self.WINDOW_NAME) / 100.0
        self.params['min_tracking_confidence'] = cv2.getTrackbarPos("Tracking Conf", self.WINDOW_NAME) / 100.0
        self.params['draw_trail'] = cv2.getTrackbarPos("Draw Trail", self.WINDOW_NAME)
        trail_duration = cv2.getTrackbarPos("Trail Dura (s)", self.WINDOW_NAME)
        self.params['trail_duration'] = max(1, trail_duration)
        self.params['pinch_threshold'] = cv2.getTrackbarPos("Pinch Thresh", self.WINDOW_NAME)


        if self.params != self.current_params:
            print("參數已變更，正在重新初始化偵測器...")
            if self.hands_detector:
                self.hands_detector.close()
            self.hands_detector = self._create_hands_detector()
            self.current_params = self.params.copy()
            
    # 移除 _draw_3d_button 方法，因為不再使用 OpenCV 繪製按鈕

    def _draw_overlays(self, frame, results): # 移除 finger_pos, is_pinching 參數
        """在影像幀上繪製所有覆蓋圖層：軌跡、標記點和UI。"""
        # 1. 繪製軌跡
        trail_duration = self.params['trail_duration']
        now = time.time()
        
        # 食指指尖位置用於軌跡
        finger_pos_for_trail = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm8 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            finger_pos_for_trail = (int(lm8.x * w), int(lm8.y * h))

        while self.trail_points and now - self.trail_points[0]['time'] > trail_duration:
            self.trail_points.popleft()

        if self.params['draw_trail'] and finger_pos_for_trail:
            self.trail_points.append({'pos': finger_pos_for_trail, 'time': now})

        if self.params['draw_trail'] and len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                point1_data = self.trail_points[i-1]
                point2_data = self.trail_points[i]
                
                pos1 = point1_data['pos']
                pos2 = point2_data['pos']
                time2 = point2_data['time']

                age_ratio = (now - time2) / trail_duration
                alpha = min(age_ratio, 1.0)
                color = (int(alpha * 255), 0, int((1 - alpha) * 255))
                cv2.line(frame, pos1, pos2, color, thickness=3)

        # 2. 繪製手部標記點
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # 3. 為捏合手勢提供視覺回饋 (僅繪製圓圈，不與按鈕互動)
        if results.multi_hand_landmarks: # 僅在檢測到手時計算捏合狀態
            hand_landmarks = results.multi_hand_landmarks[0]
            lm8 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            lm4 = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            distance = np.linalg.norm(np.array([lm8.x, lm8.y, lm8.z]) - np.array([lm4.x, lm4.y, lm4.z]))
            pinch_thresh_scaled = self.params['pinch_threshold'] / 1000.0
            
            if distance < pinch_thresh_scaled:
                h, w, _ = frame.shape
                mid_x = int(((lm8.x + lm4.x) / 2) * w)
                mid_y = int(((lm8.y + lm4.y) / 2) * h)
                cv2.circle(frame, (mid_x, mid_y), 15, (0, 255, 255), 3) # 畫一個黃色空心圓

        # 4. 繪製UI面板 (頂部深灰色條)
        cv2.rectangle(frame, (0, 0), (self.WIDTH, 50), (50, 50, 50), -1)
        
        # 不再繪製 OpenCV 離開按鈕

        return frame

    def run(self):
        """主應用程式迴圈。"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        
        if not cap.isOpened():
            print("錯誤：無法開啟攝影機。")
            return

        self._setup_gui(cap)
        self.hands_detector = self._create_hands_detector()

        while self.running and cap.isOpened():
            # 更新 Tkinter 視窗
            self.exit_app_tk.update_tkinter()
            if self.exit_app_tk.get_should_exit():
                self.running = False
                break # Tkinter 按鈕被按下，退出主迴圈

            ret, frame = cap.read()
            if not ret:
                break

            bgr_frame = cv2.flip(frame, 1).copy()
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            
            rgb_frame.flags.writeable = False
            results = self.hands_detector.process(rgb_frame)
            
            # --- 移除手部與 OpenCV 離開按鈕的互動邏輯 ---
            # 食指位置和捏合狀態仍然需要用於軌跡和捏合視覺回饋
            finger_pos = None
            is_pinching = False # 不再用於控制離開按鈕
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                lm8 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = bgr_frame.shape
                finger_pos = (int(lm8.x * w), int(lm8.y * h)) # 用於軌跡

                lm4 = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                distance = np.linalg.norm(np.array([lm8.x, lm8.y, lm8.z]) - np.array([lm4.x, lm4.y, lm4.z]))
                pinch_thresh_scaled = self.params['pinch_threshold'] / 1000.0
                if distance < pinch_thresh_scaled:
                    is_pinching = True # 用於捏合視覺回饋
            
            # 繪製所有圖層
            final_frame = self._draw_overlays(bgr_frame, results) # 移除 finger_pos, is_pinching 參數

            cv2.imshow(self.WINDOW_NAME, final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
            
            self._update_params_from_trackbars()

        print("正在離開應用程式...")
        if self.hands_detector:
            self.hands_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        self.exit_app_tk.destroy_tkinter() # 銷毀 Tkinter 視窗
        print("清理完成。")

if __name__ == "__main__":
    app = HandTrackerApp()
    app.run()