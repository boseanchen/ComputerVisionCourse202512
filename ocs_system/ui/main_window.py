"""
Improved Main Window for OCS System
改進版 CustomTkinter UI - 基於執行結果優化
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import sys
import os
from pathlib import Path

# 加入專案路徑
sys.path.append(str(Path(__file__).parent.parent))

from core.image_processor import ImageProcessor
from core.coin_classifier import CoinClassifier, CoinCounter


class OCSMainWindowV2(ctk.CTk):
    """OCS 主視窗 V2 - 改進版"""
    
    def __init__(self):
        super().__init__()
        
        # 視窗設定
        self.title("🪙 OCS - 硬幣辨識系統 V2")
        self.geometry("1600x900")
        
        # 設定主題
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 初始化核心模組
        self.processor = ImageProcessor()
        self.classifier = CoinClassifier()
        self.counter = CoinCounter()
        
        # 狀態變數
        self.current_image = None
        self.current_image_path = None
        self.result_data = None
        
        # 參數變數（優化後的預設值 - 與測試腳本一致）
        self.contrast_value = ctk.DoubleVar(value=3.0)  # 優化: 2.5 → 3.0
        self.param2_value = ctk.IntVar(value=35)        # 優化: 22 → 35 (關鍵參數)
        self.min_radius_value = ctk.IntVar(value=30)    # 保持
        self.max_radius_value = ctk.IntVar(value=95)    # 優化: 75 → 95
        
        # 建立 UI
        self._create_ui()
        
    def _create_ui(self):
        """建立使用者介面 - 左右分欄式"""
        # 主要容器 - 左右分欄 (1:3 比例)
        self.grid_columnconfigure(0, weight=1, minsize=400)  # 左側控制面板
        self.grid_columnconfigure(1, weight=3, minsize=1200)  # 右側影像區
        self.grid_rowconfigure(0, weight=1)
        
        # ========== 左側控制面板 ==========
        self.left_panel = ctk.CTkFrame(self, corner_radius=10)
        self.left_panel.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        
        # 左側分為三部分：來源選擇、結果摘要、參數調整
        self.left_panel.grid_rowconfigure(0, weight=0, minsize=200)  # 來源選擇
        self.left_panel.grid_rowconfigure(1, weight=1, minsize=300)  # 結果摘要
        self.left_panel.grid_rowconfigure(2, weight=0, minsize=350)  # 參數調整
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # 左上：影像來源選擇
        self._create_source_panel()
        
        # 左中：辨識結果摘要（合併）
        self._create_results_summary_panel()
        
        # 左下：參數調整
        self._create_parameters_panel()
        
        # ========== 右側影像顯示區 ==========
        self.right_panel = ctk.CTkFrame(self, corner_radius=10)
        self.right_panel.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        
        # 右側分為上下兩部分：原始影像、辨識結果
        self.right_panel.grid_rowconfigure(0, weight=1)  # 原始影像
        self.right_panel.grid_rowconfigure(1, weight=1)  # 辨識結果
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # 右上：原始影像
        self._create_original_image_panel()
        
        # 右下：辨識結果（疊圖）
        self._create_result_image_panel()
    
    def _create_source_panel(self):
        """建立影像來源選擇區"""
        frame = ctk.CTkFrame(self.left_panel, corner_radius=10)
        frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        
        # 標題
        title = ctk.CTkLabel(
            frame, 
            text="📷 影像來源", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(15, 10))
        
        # 選項
        self.source_var = ctk.StringVar(value="upload")
        
        upload_radio = ctk.CTkRadioButton(
            frame, text="📁 上傳圖片", variable=self.source_var,
            value="upload", font=ctk.CTkFont(size=14)
        )
        upload_radio.pack(pady=5, padx=20, anchor="w")
        
        camera_radio = ctk.CTkRadioButton(
            frame, text="📹 即時鏡頭 (開發中)", variable=self.source_var,
            value="camera", font=ctk.CTkFont(size=14), state="disabled"
        )
        camera_radio.pack(pady=5, padx=20, anchor="w")
        
        # 按鈕
        self.select_btn = ctk.CTkButton(
            frame, text="選擇圖片檔案", command=self._select_image,
            font=ctk.CTkFont(size=14), height=35
        )
        self.select_btn.pack(pady=10, padx=20, fill="x")
        
        self.recognize_btn = ctk.CTkButton(
            frame, text="🔍 開始辨識", command=self._start_recognition,
            font=ctk.CTkFont(size=16, weight="bold"), height=45,
            fg_color="green", hover_color="darkgreen", state="disabled"
        )
        self.recognize_btn.pack(pady=10, padx=20, fill="x")
        
        # 檔案路徑
        self.file_label = ctk.CTkLabel(
            frame, text="尚未選擇檔案", font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.file_label.pack(pady=(0, 10), padx=20)
    
    def _create_results_summary_panel(self):
        """建立辨識結果摘要區（合併原本的左下和右下）"""
        frame = ctk.CTkFrame(self.left_panel, corner_radius=10)
        frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # 標題
        title = ctk.CTkLabel(
            frame, text="📊 辨識結果摘要",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(15, 10))
        
        # 分隔線
        ctk.CTkFrame(frame, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=5)
        
        # 總金額（大字體）
        self.total_value_label = ctk.CTkLabel(
            frame, text="總金額: -- 元",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#4CAF50"
        )
        self.total_value_label.pack(pady=10)
        
        # 硬幣總數
        self.total_count_label = ctk.CTkLabel(
            frame, text="硬幣總數: -- 個",
            font=ctk.CTkFont(size=18)
        )
        self.total_count_label.pack(pady=5)
        
        # 詳細統計（可捲動文字框）
        self.details_textbox = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(size=13, family="Consolas"),
            wrap="word", height=150
        )
        self.details_textbox.pack(pady=10, padx=10, fill="both", expand=True)
        self.details_textbox.insert("1.0", "等待辨識...\n\n請選擇圖片並點擊「開始辨識」")
        
        # 狀態
        self.status_label = ctk.CTkLabel(
            frame, text="等待辨識...",
            font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.status_label.pack(pady=(5, 10))
    
    def _create_parameters_panel(self):
        """建立參數調整區"""
        frame = ctk.CTkFrame(self.left_panel, corner_radius=10)
        frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="nsew")
        
        # 標題
        title = ctk.CTkLabel(
            frame, text="⚙️ 參數調整",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(15, 10))
        
        # 明暗對比
        self._create_slider(
            frame, "明暗對比 (Contrast)", 
            self.contrast_value, 0.5, 5.0, 2.5, 0.1
        )
        
        # 圓形檢測閾值
        self._create_slider(
            frame, "圓形檢測閾值 (param2)",
            self.param2_value, 10, 100, 22, 1
        )
        
        # 最小半徑
        self._create_slider(
            frame, "最小半徑 (minRadius)",
            self.min_radius_value, 5, 50, 30, 1
        )
        
        # 最大半徑
        self._create_slider(
            frame, "最大半徑 (maxRadius)",
            self.max_radius_value, 50, 200, 75, 5
        )
        
        # 重置按鈕
        reset_btn = ctk.CTkButton(
            frame, text="🔄 重置為預設值",
            command=self._reset_parameters,
            font=ctk.CTkFont(size=13), height=35
        )
        reset_btn.pack(pady=10, padx=20, fill="x")
    
    def _create_slider(self, parent, label_text, variable, from_, to, default, step):
        """建立 Slider 元件"""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(pady=8, padx=20, fill="x")
        
        # 標籤與數值
        label_frame = ctk.CTkFrame(container, fg_color="transparent")
        label_frame.pack(fill="x")
        
        label = ctk.CTkLabel(
            label_frame, text=label_text,
            font=ctk.CTkFont(size=12), anchor="w"
        )
        label.pack(side="left")
        
        value_label = ctk.CTkLabel(
            label_frame, text=f"[{default}]",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#4CAF50"
        )
        value_label.pack(side="right")
        
        # Slider
        if isinstance(variable, ctk.IntVar):
            steps = int((to - from_) / step)
        else:
            steps = int((to - from_) / step)
        
        slider = ctk.CTkSlider(
            container, from_=from_, to=to,
            number_of_steps=steps,
            variable=variable,
            command=lambda v: value_label.configure(
                text=f"[{v:.1f}]" if isinstance(variable, ctk.DoubleVar) else f"[{int(v)}]"
            )
        )
        slider.pack(fill="x", pady=(5, 0))
        slider.set(default)
    
    def _create_original_image_panel(self):
        """建立原始影像顯示區（右上）"""
        frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        
        title = ctk.CTkLabel(
            frame, text="🖼️ 原始影像",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(10, 5))
        
        self.original_canvas = ctk.CTkLabel(
            frame, text="尚未載入影像",
            font=ctk.CTkFont(size=16),
            fg_color="gray20", corner_radius=10
        )
        self.original_canvas.pack(pady=10, padx=10, fill="both", expand=True)
    
    def _create_result_image_panel(self):
        """建立辨識結果影像區（右下）"""
        frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        frame.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        
        title = ctk.CTkLabel(
            frame, text="✅ 辨識結果 (疊圖)",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(10, 5))
        
        self.result_canvas = ctk.CTkLabel(
            frame, text="等待辨識結果...",
            font=ctk.CTkFont(size=16),
            fg_color="gray20", corner_radius=10
        )
        self.result_canvas.pack(pady=10, padx=10, fill="both", expand=True)
    
    # ========== 事件處理 ==========
    
    def _select_image(self):
        """選擇圖片"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("圖片檔案", "*.jpg *.jpeg *.png *.bmp"), ("所有檔案", "*.*")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_label.configure(text=f"已選擇: {os.path.basename(file_path)}")
            
            # 載入圖片
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("錯誤", "無法讀取圖片")
                return
            
            # 顯示原始影像
            self._display_image(self.current_image, self.original_canvas)
            
            # 清空結果
            self.result_canvas.configure(image=None, text="等待辨識結果...")
            
            # 啟用辨識按鈕
            self.recognize_btn.configure(state="normal")
            self.status_label.configure(
                text=f"已載入 ({self.current_image.shape[1]}x{self.current_image.shape[0]})"
            )
    
    def _start_recognition(self):
        """開始辨識"""
        if self.current_image is None:
            messagebox.showwarning("警告", "請先選擇圖片")
            return
        
        self.status_label.configure(text="辨識中...", text_color="orange")
        self.recognize_btn.configure(state="disabled")
        self.update()
        
        try:
            self._perform_recognition()
            self.status_label.configure(text="辨識完成！", text_color="green")
        except Exception as e:
            messagebox.showerror("錯誤", f"辨識失敗: {str(e)}")
            self.status_label.configure(text="辨識失敗", text_color="red")
        finally:
            self.recognize_btn.configure(state="normal")
    
    def _perform_recognition(self):
        """執行辨識（使用優化的預處理流程）"""
        # 重置計數器
        self.counter.reset()
        
        # ✅ 使用 ImageProcessor 的完整預處理 (與測試腳本一致)
        # 這會執行: 灰階 → 模糊(5,5) → CLAHE → 返回
        gray = self.processor.preprocess_image(self.current_image)
        
        # 檢測硬幣（使用調整後的參數）
        coins = self._detect_coins_with_params(gray)
        
        # 收集所有半徑（用於相對尺寸分類）
        all_radii = [coin['radius'] for coin in coins]
        
        # 分類硬幣
        results = []
        for i, coin in enumerate(coins):
            roi = self.processor.extract_coin_roi(
                self.current_image, coin['x'], coin['y'], coin['radius']
            )
            color_features = self.processor.extract_color_features(roi)
            
            # 傳入所有半徑以進行相對尺寸分類
            classification = self.classifier.classify_coin(
                roi, coin['radius'], color_features, all_radii
            )
            
            self.counter.add_coin(classification['denomination'], classification['side'])
            
            results.append({
                'id': i + 1, 'x': coin['x'], 'y': coin['y'],
                'radius': coin['radius'],
                'denomination': classification['denomination'],
                'side': classification['side'],
                'confidence': classification['confidence']
            })
        
        # 更新顯示
        self._update_results(results)
    
    def _apply_contrast(self, image, clip_limit):
        """應用對比度增強"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def _detect_coins_with_params(self, gray_image):
        """使用當前參數檢測硬幣（優化版）"""
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)  # 優化: (5,5) → (9,9)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, 
            minDist=80,  # 優化: 30 → 80 (避免重複檢測)
            param1=60,   # 優化: 50 → 60
            param2=self.param2_value.get(),
            minRadius=self.min_radius_value.get(),
            maxRadius=self.max_radius_value.get()
        )
        
        coins = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle
                coins.append({'x': int(x), 'y': int(y), 'radius': int(radius)})
        
        return coins
    
    def _update_results(self, results):
        """更新結果顯示"""
        stats = self.counter.get_statistics()
        
        # 更新摘要
        self.total_value_label.configure(text=f"總金額: {stats['total_value']} 元")
        self.total_count_label.configure(text=f"硬幣總數: {stats['total_count']} 個")
        
        # 更新詳細資訊
        details = "=" * 40 + "\n硬幣辨識詳細結果\n" + "=" * 40 + "\n\n"
        for denom in [50, 10, 5, 1]:
            data = stats['breakdown'][denom]
            if data['total'] > 0:
                details += f"【{denom}元硬幣】\n"
                details += f"  總數: {data['total']} 個\n"
                details += f"  正面: {data['heads']} 個 / 反面: {data['tails']} 個\n"
                details += f"  小計: {denom * data['total']} 元\n\n"
        
        self.details_textbox.delete("1.0", "end")
        self.details_textbox.insert("1.0", details)
        
        # 繪製並顯示結果
        result_image = self._draw_results(self.current_image.copy(), results)
        self._display_image(result_image, self.result_canvas)
    
    def _draw_results(self, image, results):
        """繪製辨識結果"""
        colors = {
            1: (100, 150, 255),   # 淺藍
            5: (255, 200, 100),   # 金黃
            10: (255, 150, 50),   # 橙色
            50: (150, 255, 150)   # 淺綠
        }
        
        for coin in results:
            x, y, r = coin['x'], coin['y'], coin['radius']
            denom = coin['denomination']
            side = coin['side']
            color = colors.get(denom, (255, 255, 255))
            
            cv2.circle(image, (x, y), r, color, 3)
            label = f"{denom}$ {side[0].upper()}"
            cv2.putText(image, label, (x - 30, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(image, f"#{coin['id']}", (x - 10, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def _display_image(self, cv_image, canvas_widget):
        """統一影像顯示（保持尺寸比例一致）"""
        # 固定顯示尺寸
        target_width = 1100
        target_height = 380
        
        # 轉換顏色
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 調整大小（保持比例）
        h, w = rgb.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))
        
        # 建立固定大小畫布（置中）
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 顯示
        pil_image = Image.fromarray(canvas)
        ctk_image = ctk.CTkImage(
            light_image=pil_image, dark_image=pil_image,
            size=(target_width, target_height)
        )
        canvas_widget.configure(image=ctk_image, text="")
        canvas_widget.image = ctk_image
    
    def _reset_parameters(self):
        """重置參數為預設值（優化後 - 與測試腳本一致）"""
        self.contrast_value.set(3.0)   # 優化值
        self.param2_value.set(35)      # 優化值
        self.min_radius_value.set(30)  # 保持
        self.max_radius_value.set(95)  # 優化值
        messagebox.showinfo("提示", "參數已重置為優化後的預設值 (param2=35)")


def main():
    """主程式"""
    app = OCSMainWindowV2()
    app.mainloop()


if __name__ == "__main__":
    main()
