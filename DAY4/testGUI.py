"""
浮動圓形按鈕 GUI + MediaPipe 手勢辨識
- 預設為右下角的小圓形按鈕，顯示手勢簡稱
- 點擊展開為正方形設定視窗，顯示 webcam 畫面
- 始終置頂

手勢動作 (Alt+Tab 視窗切換):
- 握拳 👊 = 啟動 Alt+Tab 視窗切換
- 無手勢 = 動作間的斷點 (準備下一個動作)
- 拇指向上 👍 = 上一個視窗 (Shift+Tab)
- 拇指向下 👎 = 下一個視窗 (Tab)
- 再次握拳 👊 = 確認選擇並關閉 Alt+Tab

安裝套件:
    pip install pyautogui mediapipe customtkinter opencv-python pillow

需要先下載手勢辨識模型:
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
放到 DAY4 資料夾 (會自動下載)
"""

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
import os
import urllib.request
import time
from collections import deque

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 鍵盤控制
import pyautogui

# 設定外觀
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# 尺寸設定
CIRCLE_SIZE = 70          # 圓形按鈕直徑
EXPANDED_SIZE = 500       # 展開後的正方形邊長 (加大以顯示事件)
MARGIN = 20               # 距離螢幕邊緣的距離

# 模型路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "gesture_recognizer.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"

# 手勢對照表
GESTURE_MAP = {
    "None": {"full": "無手勢", "short": "---"},
    "Closed_Fist": {"full": "握拳 👊", "short": "👊"},
    "Open_Palm": {"full": "張開手掌 🖐️", "short": "🖐️"},
    "Pointing_Up": {"full": "指向上 ☝️", "short": "☝️"},
    "Thumb_Down": {"full": "拇指向下 👎", "short": "👎"},
    "Thumb_Up": {"full": "拇指向上 👍", "short": "👍"},
    "Victory": {"full": "勝利 ✌️", "short": "✌️"},
    "ILoveYou": {"full": "我愛你 🤟", "short": "🤟"},
}

# 動作事件對照表
ACTION_MAP = {
    "alt_tab_start": {"full": "🔄 Alt+Tab 啟動", "color": (0, 255, 255)},
    "prev_window": {"full": "👍 上一個視窗", "color": (0, 255, 0)},
    "next_window": {"full": "👎 下一個視窗", "color": (255, 128, 0)},
    "confirm_select": {"full": "👊 確認選擇", "color": (0, 128, 255)},
}

# pyautogui 設定
pyautogui.FAILSAFE = False  # 關閉安全模式 (避免滑鼠移到角落時中斷)


def download_model():
    """下載手勢辨識模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"正在下載手勢辨識模型...")
        print(f"URL: {MODEL_URL}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"模型已下載至: {MODEL_PATH}")
        except Exception as e:
            print(f"下載失敗: {e}")
            print("請手動下載模型並放到 DAY4 資料夾")
            return False
    return True


class FloatingBubble(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 狀態
        self.is_expanded = False
        self.webcam_running = False
        self.cap = None
        self.current_gesture = "None"
        self.prev_gesture = "None"           # 上一幀的手勢
        self.gesture_recognizer = None

        # 動作檢測狀態
        self.current_action = None           # 當前觸發的動作
        self.action_display_time = 0         # 動作顯示時間
        self.action_display_duration = 1.0   # 動作顯示持續時間

        # Alt+Tab 狀態機
        self.alt_tab_active = False          # Alt+Tab 是否啟動中
        self.ready_for_action = False        # 是否準備好接收下一個動作 (經過 None 手勢)

        # 事件歷史記錄
        self.event_history = deque(maxlen=5)  # 最近 5 個事件

        # 移除標題欄
        self.overrideredirect(True)

        # 置頂
        self.attributes('-topmost', True)

        # 設定背景
        self.configure(fg_color='#1a1a2e')

        # 取得螢幕尺寸
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # 初始化手勢辨識器
        self.init_gesture_recognizer()

        # 初始化為圓形模式
        self.setup_circle_mode()

        # 拖曳功能
        self._drag_x = 0
        self._drag_y = 0

        # 啟動 webcam
        self.start_webcam()

    def init_gesture_recognizer(self):
        """初始化手勢辨識器"""
        if not os.path.exists(MODEL_PATH):
            print(f"找不到模型: {MODEL_PATH}")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=2
            )
            self.gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
            print("手勢辨識器初始化成功")
        except Exception as e:
            print(f"手勢辨識器初始化失敗: {e}")

    def process_gesture_state_machine(self, current_time):
        """
        手勢狀態機處理 Alt+Tab 視窗切換

        流程:
        1. 握拳 (Closed_Fist) → 啟動 Alt+Tab
        2. 無手勢 (None) → 準備接收下一個動作
        3. 拇指向上 (Thumb_Up) → 上一個視窗
        4. 拇指向下 (Thumb_Down) → 下一個視窗
        5. 再次握拳 (Closed_Fist) → 確認選擇，關閉 Alt+Tab
        """
        gesture = self.current_gesture
        prev = self.prev_gesture

        # 檢測手勢變化 (從其他手勢變成當前手勢)
        gesture_changed = (gesture != prev)

        if not gesture_changed:
            return

        # ===== 狀態機邏輯 =====

        # 狀態 1: Alt+Tab 未啟動
        if not self.alt_tab_active:
            # 握拳 → 啟動 Alt+Tab
            if gesture == "Closed_Fist" and prev == "None":
                self.start_alt_tab(current_time)
                self.ready_for_action = False

        # 狀態 2: Alt+Tab 已啟動
        else:
            # 變成無手勢 → 準備接收動作
            if gesture == "None":
                self.ready_for_action = True
                print("準備接收下一個動作...")

            # 從無手勢變成其他手勢 → 執行動作
            elif self.ready_for_action and prev == "None":
                if gesture == "Thumb_Up":
                    # 上一個視窗
                    self.switch_prev_window(current_time)
                    self.ready_for_action = False

                elif gesture == "Thumb_Down":
                    # 下一個視窗
                    self.switch_next_window(current_time)
                    self.ready_for_action = False

                elif gesture == "Closed_Fist":
                    # 確認選擇
                    self.confirm_selection(current_time)
                    self.ready_for_action = False

    def start_alt_tab(self, current_time):
        """啟動 Alt+Tab"""
        print("啟動 Alt+Tab")
        self.alt_tab_active = True

        # 按下 Alt 並按一次 Tab
        pyautogui.keyDown('alt')
        pyautogui.press('tab')

        self.trigger_action("alt_tab_start", current_time)

    def switch_prev_window(self, current_time):
        """切換到上一個視窗 (Shift+Tab)"""
        if not self.alt_tab_active:
            return

        print("上一個視窗")
        pyautogui.hotkey('shift', 'tab')
        self.trigger_action("prev_window", current_time)

    def switch_next_window(self, current_time):
        """切換到下一個視窗 (Tab)"""
        if not self.alt_tab_active:
            return

        print("下一個視窗")
        pyautogui.press('tab')
        self.trigger_action("next_window", current_time)

    def confirm_selection(self, current_time):
        """確認選擇並關閉 Alt+Tab"""
        print("確認選擇")

        # 放開 Alt 鍵
        pyautogui.keyUp('alt')
        self.alt_tab_active = False

        self.trigger_action("confirm_select", current_time)

    def trigger_action(self, action_type, current_time):
        """觸發動作事件"""
        self.current_action = action_type
        self.action_display_time = current_time

        action_info = ACTION_MAP.get(action_type, {})
        event_text = f"{action_info.get('full', action_type)}"

        # 加入事件歷史
        time_str = time.strftime("%H:%M:%S")
        self.event_history.append(f"[{time_str}] {event_text}")

        print(f"觸發動作: {event_text}")

    def setup_circle_mode(self):
        """設定圓形按鈕模式"""
        self.is_expanded = False

        # 清除所有子元件
        for widget in self.winfo_children():
            widget.destroy()

        # 設定視窗大小和位置 (右下角)
        x = self.screen_width - CIRCLE_SIZE - MARGIN
        y = self.screen_height - CIRCLE_SIZE - MARGIN - 40
        self.geometry(f"{CIRCLE_SIZE}x{CIRCLE_SIZE}+{x}+{y}")

        # 建立圓形按鈕容器
        self.circle_frame = ctk.CTkFrame(
            self,
            width=CIRCLE_SIZE,
            height=CIRCLE_SIZE,
            corner_radius=CIRCLE_SIZE // 2,
            fg_color='#4a90d9'
        )
        self.circle_frame.pack(expand=True, fill='both')
        self.circle_frame.pack_propagate(False)

        # 圓形按鈕上的手勢顯示
        self.circle_label = ctk.CTkLabel(
            self.circle_frame,
            text="---",
            font=ctk.CTkFont(size=28),
            text_color='white'
        )
        self.circle_label.place(relx=0.5, rely=0.5, anchor='center')

        # 綁定事件
        self.circle_frame.bind('<Button-1>', self.on_click)
        self.circle_label.bind('<Button-1>', self.on_click)
        self.circle_frame.bind('<ButtonPress-1>', self.start_drag)
        self.circle_frame.bind('<B1-Motion>', self.on_drag)
        self.circle_label.bind('<ButtonPress-1>', self.start_drag)
        self.circle_label.bind('<B1-Motion>', self.on_drag)

        # Hover 效果
        self.circle_frame.bind('<Enter>', lambda e: self.circle_frame.configure(fg_color='#5ba0e9'))
        self.circle_frame.bind('<Leave>', lambda e: self.circle_frame.configure(fg_color='#4a90d9'))

    def setup_expanded_mode(self):
        """設定展開後的正方形視窗模式"""
        self.is_expanded = True

        # 清除所有子元件
        for widget in self.winfo_children():
            widget.destroy()

        # 設定視窗大小和位置
        x = self.screen_width - EXPANDED_SIZE - MARGIN
        y = self.screen_height - EXPANDED_SIZE - MARGIN - 40
        self.geometry(f"{EXPANDED_SIZE}x{EXPANDED_SIZE}+{x}+{y}")

        # 主容器
        self.main_frame = ctk.CTkFrame(
            self,
            corner_radius=15,
            fg_color='#1a1a2e'
        )
        self.main_frame.pack(expand=True, fill='both', padx=2, pady=2)

        # 標題欄
        self.title_bar = ctk.CTkFrame(
            self.main_frame,
            height=40,
            corner_radius=0,
            fg_color='#2d2d44'
        )
        self.title_bar.pack(fill='x', padx=10, pady=(10, 5))
        self.title_bar.pack_propagate(False)

        # 標題文字
        self.title_label = ctk.CTkLabel(
            self.title_bar,
            text="🖐️ 手勢辨識 (Alt+Tab)",
            font=ctk.CTkFont(size=14, weight='bold')
        )
        self.title_label.pack(side='left', padx=10, pady=5)

        # 關閉按鈕
        self.close_btn = ctk.CTkButton(
            self.title_bar,
            text="✕",
            width=30,
            height=30,
            corner_radius=15,
            fg_color='transparent',
            hover_color='#ff6b6b',
            command=self.collapse
        )
        self.close_btn.pack(side='right', padx=5, pady=5)

        # 內容區域
        self.content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color='transparent'
        )
        self.content_frame.pack(expand=True, fill='both', padx=10, pady=5)

        # Webcam 顯示區域
        self.video_label = ctk.CTkLabel(
            self.content_frame,
            text="Webcam Loading...",
            width=EXPANDED_SIZE - 40,
            height=200
        )
        self.video_label.pack(pady=5)

        # 手勢顯示
        self.gesture_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color='#2d2d44',
            corner_radius=10
        )
        self.gesture_frame.pack(fill='x', pady=5)

        self.gesture_label = ctk.CTkLabel(
            self.gesture_frame,
            text="手勢: ---",
            font=ctk.CTkFont(size=18, weight='bold'),
            text_color='#00ff7f'
        )
        self.gesture_label.pack(pady=10)

        # 動作顯示區域
        self.action_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color='#3d3d5c',
            corner_radius=10
        )
        self.action_frame.pack(fill='x', pady=5)

        self.action_label = ctk.CTkLabel(
            self.action_frame,
            text="動作: ---",
            font=ctk.CTkFont(size=16, weight='bold'),
            text_color='#ffff00'
        )
        self.action_label.pack(pady=8)

        # 事件歷史區域
        self.history_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color='#252540',
            corner_radius=10
        )
        self.history_frame.pack(fill='x', pady=5)

        history_title = ctk.CTkLabel(
            self.history_frame,
            text="事件歷史",
            font=ctk.CTkFont(size=12),
            text_color='#888888'
        )
        history_title.pack(anchor='w', padx=10, pady=(5, 0))

        self.history_text = ctk.CTkLabel(
            self.history_frame,
            text="尚無事件",
            font=ctk.CTkFont(size=11),
            text_color='#aaaaaa',
            justify='left',
            anchor='w'
        )
        self.history_text.pack(fill='x', padx=10, pady=(0, 5))

        # 控制區域
        control_frame = ctk.CTkFrame(self.content_frame, fg_color='transparent')
        control_frame.pack(fill='x', pady=5)

        # Webcam 開關
        self.webcam_var = ctk.BooleanVar(value=self.webcam_running)
        self.webcam_switch = ctk.CTkSwitch(
            control_frame,
            text="Webcam",
            variable=self.webcam_var,
            command=self.toggle_webcam
        )
        self.webcam_switch.pack(side='left', padx=10)

        # 置頂開關
        self.topmost_var = ctk.BooleanVar(value=True)
        self.topmost_switch = ctk.CTkSwitch(
            control_frame,
            text="置頂",
            variable=self.topmost_var,
            command=self.toggle_topmost
        )
        self.topmost_switch.pack(side='right', padx=10)

        # 綁定標題欄拖曳
        self.title_bar.bind('<ButtonPress-1>', self.start_drag)
        self.title_bar.bind('<B1-Motion>', self.on_drag)
        self.title_label.bind('<ButtonPress-1>', self.start_drag)
        self.title_label.bind('<B1-Motion>', self.on_drag)

    def start_webcam(self):
        """啟動 webcam"""
        if self.webcam_running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("無法開啟攝影機")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.webcam_running = True
        print("Webcam 已啟動")

        # 啟動更新執行緒
        self.update_thread = threading.Thread(target=self.webcam_loop, daemon=True)
        self.update_thread.start()

    def stop_webcam(self):
        """停止 webcam"""
        self.webcam_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Webcam 已停止")

    def toggle_webcam(self):
        """切換 webcam 開關"""
        if self.webcam_var.get():
            self.start_webcam()
        else:
            self.stop_webcam()
            self.current_gesture = "None"
            self.update_gesture_display()

    def webcam_loop(self):
        """Webcam 處理迴圈"""
        while self.webcam_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()

            # 水平翻轉
            frame = cv2.flip(frame, 1)

            # 轉換為 RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 手勢辨識
            if self.gesture_recognizer is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                try:
                    result = self.gesture_recognizer.recognize(mp_image)

                    if result.gestures and len(result.gestures) > 0:
                        gesture = result.gestures[0][0].category_name
                        self.current_gesture = gesture
                    else:
                        self.current_gesture = "None"
                except Exception as e:
                    pass

            # 處理手勢狀態機 (Alt+Tab 控制)
            self.process_gesture_state_machine(current_time)

            # 更新上一幀手勢
            self.prev_gesture = self.current_gesture

            # 檢查動作顯示是否過期
            if self.current_action and (current_time - self.action_display_time) > self.action_display_duration:
                self.current_action = None

            # 更新 UI (在主執行緒)
            self.after(0, self.update_gesture_display)
            self.after(0, self.update_action_display)

            # 如果展開模式，更新影像
            if self.is_expanded:
                self.after(0, lambda f=frame.copy(): self.update_video_display(f))

            # 控制幀率
            cv2.waitKey(30)

    def update_gesture_display(self):
        """更新手勢顯示"""
        gesture_info = GESTURE_MAP.get(self.current_gesture, GESTURE_MAP["None"])

        # 更新圓形按鈕 (如果存在)
        if hasattr(self, 'circle_label') and self.circle_label.winfo_exists():
            # 如果有動作觸發，顯示動作
            if self.current_action:
                short_action = {
                    "alt_tab_start": "🔄",
                    "prev_window": "👍",
                    "next_window": "👎",
                    "confirm_select": "✅"
                }.get(self.current_action, "---")
                self.circle_label.configure(text=short_action)
            elif self.alt_tab_active:
                # Alt+Tab 啟動中，顯示狀態
                self.circle_label.configure(text="🔄")
            else:
                self.circle_label.configure(text=gesture_info["short"])

        # 更新圓形按鈕顏色 (根據 Alt+Tab 狀態)
        if hasattr(self, 'circle_frame') and self.circle_frame.winfo_exists():
            if self.alt_tab_active:
                self.circle_frame.configure(fg_color='#e67e22')  # 橘色表示啟動中
            else:
                self.circle_frame.configure(fg_color='#4a90d9')  # 預設藍色

        # 更新展開視窗 (如果存在)
        if hasattr(self, 'gesture_label') and self.gesture_label.winfo_exists():
            status = " [Alt+Tab 啟動中]" if self.alt_tab_active else ""
            self.gesture_label.configure(text=f"手勢: {gesture_info['full']}{status}")

    def update_action_display(self):
        """更新動作顯示"""
        # 更新動作標籤
        if hasattr(self, 'action_label') and self.action_label.winfo_exists():
            if self.current_action:
                action_info = ACTION_MAP.get(self.current_action, {})
                self.action_label.configure(
                    text=f"動作: {action_info.get('full', '---')}",
                    text_color='#00ff00'
                )
            else:
                self.action_label.configure(
                    text="動作: ---",
                    text_color='#ffff00'
                )

        # 更新事件歷史
        if hasattr(self, 'history_text') and self.history_text.winfo_exists():
            if self.event_history:
                history_str = "\n".join(list(self.event_history)[-5:])
                self.history_text.configure(text=history_str)
            else:
                self.history_text.configure(text="尚無事件")

    def update_video_display(self, frame):
        """更新影像顯示"""
        if not hasattr(self, 'video_label') or not self.video_label.winfo_exists():
            return

        h, w = frame.shape[:2]

        # 繪製手勢文字
        gesture_info = GESTURE_MAP.get(self.current_gesture, GESTURE_MAP["None"])
        cv2.putText(
            frame,
            gesture_info["full"],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # 繪製 Alt+Tab 狀態
        if self.alt_tab_active:
            cv2.putText(
                frame,
                "[Alt+Tab Active]",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # 繪製提示
            tips = "Thumb Up=Prev | Thumb Down=Next | Fist=Confirm"
            cv2.putText(
                frame,
                tips,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

        # 繪製動作文字 (如果有)
        if self.current_action:
            action_info = ACTION_MAP.get(self.current_action, {})
            action_text = action_info.get("full", "")
            action_color = action_info.get("color", (255, 255, 255))

            # 繪製半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # 繪製動作文字
            text_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2 + text_size[1] // 2
            cv2.putText(
                frame,
                action_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                action_color,
                2
            )

        # 調整大小
        frame = cv2.resize(frame, (EXPANDED_SIZE - 40, 200))

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 轉換為 CTkImage
        image = Image.fromarray(frame_rgb)
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=(EXPANDED_SIZE - 40, 200))

        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def on_click(self, event):
        """點擊圓形按鈕時展開"""
        if not hasattr(self, '_click_x'):
            self.expand()
            return

        dx = abs(event.x_root - self._click_x)
        dy = abs(event.y_root - self._click_y)

        if dx < 5 and dy < 5:
            self.expand()

    def expand(self):
        """展開視窗"""
        if not self.is_expanded:
            self.setup_expanded_mode()
            # 更新 webcam 開關狀態
            if hasattr(self, 'webcam_var'):
                self.webcam_var.set(self.webcam_running)

    def collapse(self):
        """收合視窗"""
        if self.is_expanded:
            self.setup_circle_mode()

    def start_drag(self, event):
        """開始拖曳"""
        self._drag_x = event.x
        self._drag_y = event.y
        self._click_x = event.x_root
        self._click_y = event.y_root

    def on_drag(self, event):
        """拖曳中"""
        x = self.winfo_x() + event.x - self._drag_x
        y = self.winfo_y() + event.y - self._drag_y
        self.geometry(f"+{x}+{y}")

    def toggle_topmost(self):
        """切換置頂狀態"""
        is_top = self.topmost_var.get()
        self.attributes('-topmost', is_top)

    def on_closing(self):
        """關閉視窗"""
        # 確保釋放 Alt 鍵
        if self.alt_tab_active:
            pyautogui.keyUp('alt')
            self.alt_tab_active = False
        self.stop_webcam()
        self.destroy()


def main():
    # 下載模型
    if not download_model():
        print("請手動下載模型後再執行")
        print(f"下載網址: {MODEL_URL}")

    app = FloatingBubble()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
