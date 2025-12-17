"""
Image Processor Module
處理影像預處理、硬幣檢測等核心功能
使用 Contour Detection + HoughCircles 方法
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


class ImageProcessor:
    """影像處理器 - 負責硬幣檢測與特徵提取"""
    
    def __init__(self):
        """初始化影像處理器"""
        self.debug_mode = False
        self.target_width = 1920  # 標準解析度寬度
    
    def resize_to_standard(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        將高解析度圖片縮放到標準解析度
        
        Args:
            image: 原始圖片
            
        Returns:
            resized_image: 縮放後的圖片
            scale: 縮放比例 (用於座標還原)
        """
        h, w = image.shape[:2]
        if w > self.target_width:
            scale = self.target_width / w
            new_w = self.target_width
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, scale
        return image, 1.0
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        影像預處理
        
        Args:
            image: 原始 BGR 影像
            
        Returns:
            處理後的灰階影像
        """
        # 轉換為灰階
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 對比度增強 (CLAHE) - 提高 clipLimit 以增強對比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_coins_contours(self, image: np.ndarray) -> List[Dict]:
        """
        使用 Contour Detection 檢測硬幣
        
        Args:
            image: 原始 BGR 影像
            
        Returns:
            硬幣資訊列表 [{x, y, radius, contour}, ...]
        """
        # 預處理
        gray = self.preprocess_image(image)
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形態學操作 - 閉運算填補空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        coins = []
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            # 計算面積
            area = cv2.contourArea(contour)
            
            # 過濾太小或太大的輪廓 (雜訊或異常)
            if area < 800 or area > image_area * 0.3:  # 稍微放寬 (1000 → 800)
                continue
            
            # 計算最小外接圓
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # 半徑範圍過濾
            if radius < 20 or radius > 150:
                continue
            
            # 計算圓形度 (circularity)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # 圓形度門檻 (稍微放寬)
            if circularity < 0.80:  # 0.85 → 0.80
                continue
            
            # 長寬比檢查 (圓形應接近 1:1)
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue
            
            # 只保留接近圓形的物體 (硬幣)
            coins.append({
                'x': int(x),
                'y': int(y),
                'radius': int(radius),
                'area': area,
                'contour': contour,
                'circularity': circularity
            })
        
        return coins
    
    def detect_coins_hough(self, image: np.ndarray) -> List[Dict]:
        """
        使用 HoughCircles 檢測硬幣
        
        Args:
            image: 原始 BGR 影像
            
        Returns:
            硬幣資訊列表 [{x, y, radius}, ...]
        """
        # 預處理
        gray = self.preprocess_image(image)
        
        # 額外的高斯模糊（重要！）
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Hough Circle Transform (優化後生產環境參數)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=80,      # 硬幣之間的最小距離
            param1=60,       # Canny 邊緣檢測高閾值
            param2=35,       # 圓心檢測閾值 (生產環境推薦值)
            minRadius=30,    # 最小半徑
            maxRadius=95     # 最大半徑
        )
        
        coins = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle
                coins.append({
                    'x': int(x),
                    'y': int(y),
                    'radius': int(radius)
                })
        
        return coins
    
    def detect_coins_hybrid(self, image: np.ndarray) -> List[Dict]:
        """
        混合方法：結合 Contour 和 HoughCircles
        
        Args:
            image: 原始 BGR 影像
            
        Returns:
            硬幣資訊列表
        """
        # 先用 Contour 檢測
        coins_contour = self.detect_coins_contours(image)
        
        # 再用 HoughCircles 驗證/補充
        coins_hough = self.detect_coins_hough(image)
        
        # 合併結果 (去重)
        # 這裡簡化處理，優先使用 Contour 結果
        return coins_contour if len(coins_contour) > 0 else coins_hough
    
    def extract_coin_roi(self, image: np.ndarray, x: int, y: int, radius: int, 
                         padding: float = 1.2) -> np.ndarray:
        """
        提取硬幣 ROI (Region of Interest)
        
        Args:
            image: 原始影像
            x, y: 硬幣中心座標
            radius: 硬幣半徑
            padding: 擴展係數
            
        Returns:
            硬幣 ROI 影像
        """
        # 計算 ROI 範圍 (加上 padding)
        r = int(radius * padding)
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(image.shape[1], x + r)
        y2 = min(image.shape[0], y + r)
        
        # 提取 ROI
        roi = image[y1:y2, x1:x2]
        
        return roi
    
    def extract_color_features(self, roi: np.ndarray) -> Dict:
        """
        提取硬幣顏色特徵 (改進版 - 使用中心區域)
        
        Args:
            roi: 硬幣 ROI 影像
            
        Returns:
            顏色特徵字典 {mean_hue, mean_saturation, mean_value, is_golden, is_silver}
        """
        # 轉換到 HSV 色彩空間
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 只分析中心區域 (避免邊緣干擾)
        h, w = roi.shape[:2]
        if h > 10 and w > 10:  # 確保 ROI 足夠大
            center_roi = hsv[h//4:3*h//4, w//4:3*w//4]
        else:
            center_roi = hsv
        
        # 計算 HSV 平均值
        mean_h = np.mean(center_roi[:, :, 0])
        mean_s = np.mean(center_roi[:, :, 1])
        mean_v = np.mean(center_roi[:, :, 2])
        
        # 計算 BGR 平均值（輔助判斷）
        mean_b = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_r = np.mean(roi[:, :, 2])
        
        # 判斷金色（10元、5元）- 調整範圍
        is_golden_hue = 15 < mean_h < 35  # 黃色範圍
        is_golden_saturation = mean_s > 40  # 飽和度要求
        is_golden_value = mean_v > 80  # 亮度要求
        
        # 判斷銀色（50元、1元）- 高亮度、低飽和度
        is_silver = (mean_s < 40) and (mean_v > 100)
        
        # 綜合判斷金色（排除銀色）
        is_golden = is_golden_hue and is_golden_saturation and is_golden_value and not is_silver
        
        return {
            'mean_hue': mean_h,
            'mean_saturation': mean_s,
            'mean_value': mean_v,
            'mean_bgr': (mean_b, mean_g, mean_r),
            'is_golden': is_golden,
            'is_silver': is_silver
        }
    
    def draw_coins(self, image: np.ndarray, coins: List[Dict], 
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        在影像上繪製檢測到的硬幣
        
        Args:
            image: 原始影像
            coins: 硬幣列表
            color: 繪製顏色 (BGR)
            
        Returns:
            繪製後的影像
        """
        result = image.copy()
        
        for coin in coins:
            x, y, radius = coin['x'], coin['y'], coin['radius']
            
            # 繪製圓形
            cv2.circle(result, (x, y), radius, color, 2)
            
            # 繪製中心點
            cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
            
            # 標註半徑資訊
            cv2.putText(result, f"R:{radius}", (x - 20, y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result


if __name__ == "__main__":
    # 測試程式碼
    processor = ImageProcessor()
    
    # 載入測試影像
    test_image_path = "../assets/test_images/sample.jpg"
    # image = cv2.imread(test_image_path)
    # coins = processor.detect_coins_hybrid(image)
    # print(f"檢測到 {len(coins)} 個硬幣")
