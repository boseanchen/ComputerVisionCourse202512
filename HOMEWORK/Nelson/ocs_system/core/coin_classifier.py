"""
Improved Coin Classifier Module
優化版硬幣分類器 - 基於實際測試結果改進
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List


class CoinClassifierV2:
    """硬幣分類器 V2 - 優化版"""
    
    # 台幣硬幣實際規格 (mm)
    COIN_SPECS = {
        1: {'diameter': 20.0, 'color': 'golden_light', 'material': 'aluminum'},
        5: {'diameter': 22.0, 'color': 'golden', 'material': 'brass'},
        10: {'diameter': 26.0, 'color': 'golden_dark', 'material': 'bronze'},
        50: {'diameter': 28.0, 'color': 'silver', 'material': 'cupronickel'}
    }
    
    def __init__(self):
        """初始化分類器"""
        self.reference_diameter = None
        
    def classify_denomination_improved(self, radius: int, color_features: Dict, 
                                      all_radii: List[int] = None) -> int:
        """
        改進的面額分類 - 使用相對尺寸
        
        Args:
            radius: 硬幣半徑 (像素)
            color_features: 顏色特徵
            all_radii: 所有檢測到的硬幣半徑列表（用於相對大小計算）
            
        Returns:
            面額 (1, 5, 10, 50)
        """
        is_golden = color_features.get('is_golden', False)
        is_silver = color_features.get('is_silver', False)
        
        # 計算相對大小
        if all_radii and len(all_radii) > 1:
            max_radius = max(all_radii)
            min_radius = min(all_radii)
            
            # 避免除以零
            if max_radius > min_radius:
                # 歸一化到 0-1 範圍
                relative_size = (radius - min_radius) / (max_radius - min_radius)
            else:
                relative_size = 0.5
        else:
            # 使用絕對半徑判斷
            relative_size = None
        
        # 優先使用顏色判斷
        if is_silver:
            # 銀色 → 一定是 50元
            return 50
        
        # 金色系硬幣: 1元, 5元, 10元
        if is_golden or not is_silver:
            if relative_size is not None:
                # 使用相對尺寸分類
                if relative_size < 0.30:  # 最小的
                    return 1  # 1元 (20mm)
                elif relative_size < 0.65:  # 中等
                    return 5  # 5元 (22mm)
                else:  # 較大的金色
                    return 10  # 10元 (26mm)
            else:
                # 使用絕對半徑分類（備用方案）
                if radius < 30:
                    return 1
                elif radius < 38:
                    return 5
                else:
                    return 10
        
        # 預設返回（不應該到這裡）
        return 10
    
    def classify_side(self, roi: np.ndarray, denomination: int) -> str:
        """
        辨識硬幣正反面 (使用紋理特徵分析)
        
        Args:
            roi: 硬幣 ROI 影像
            denomination: 硬幣面額
            
        Returns:
            'heads' (正面) 或 'tails' (反面)
        """
        # 轉換為灰階
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # 計算紋理複雜度
        texture_score = self._calculate_texture_complexity(gray)
        
        # 根據紋理複雜度判斷
        # 正面（人像）通常紋理較複雜
        threshold = 0.45  # 降低閾值，更容易判斷為正面
        
        if texture_score > threshold:
            return 'heads'  # 正面
        else:
            return 'tails'  # 反面
    
    def _calculate_texture_complexity(self, gray: np.ndarray) -> float:
        """
        計算紋理複雜度
        
        使用多種方法:
        1. 邊緣密度
        2. 標準差
        3. 梯度強度
        
        Args:
            gray: 灰階影像
            
        Returns:
            紋理複雜度分數 (0-1)
        """
        # 1. 邊緣密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. 標準差 (對比度)
        std_dev = np.std(gray) / 255.0
        
        # 3. 梯度強度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_score = np.mean(gradient_magnitude) / 255.0
        
        # 綜合分數 (加權平均)
        complexity = (edge_density * 0.4 + std_dev * 0.3 + gradient_score * 0.3)
        
        return complexity
    
    def classify_coin(self, roi: np.ndarray, radius: int, 
                     color_features: Dict, all_radii: List[int] = None) -> Dict:
        """
        完整硬幣分類 (面額 + 正反面)
        
        Args:
            roi: 硬幣 ROI
            radius: 硬幣半徑
            color_features: 顏色特徵
            all_radii: 所有硬幣半徑列表
            
        Returns:
            分類結果 {denomination, side, confidence}
        """
        # 辨識面額
        denomination = self.classify_denomination_improved(radius, color_features, all_radii)
        
        # 辨識正反面
        side = self.classify_side(roi, denomination)
        
        # 計算信心度（簡化版）
        confidence = 0.85
        
        return {
            'denomination': denomination,
            'side': side,
            'confidence': confidence
        }


class CoinCounter:
    """硬幣計數與統計"""
    
    def __init__(self):
        """初始化計數器"""
        self.coins_data = []
    
    def add_coin(self, denomination: int, side: str):
        """
        新增硬幣資料
        
        Args:
            denomination: 面額
            side: 正反面
        """
        self.coins_data.append({
            'denomination': denomination,
            'side': side
        })
    
    def get_statistics(self) -> Dict:
        """
        獲取統計資料
        
        Returns:
            統計結果 {total_value, total_count, breakdown}
        """
        # 統計各面額數量
        breakdown = {
            1: {'total': 0, 'heads': 0, 'tails': 0},
            5: {'total': 0, 'heads': 0, 'tails': 0},
            10: {'total': 0, 'heads': 0, 'tails': 0},
            50: {'total': 0, 'heads': 0, 'tails': 0}
        }
        
        for coin in self.coins_data:
            denom = coin['denomination']
            side = coin['side']
            
            breakdown[denom]['total'] += 1
            if side == 'heads':
                breakdown[denom]['heads'] += 1
            else:
                breakdown[denom]['tails'] += 1
        
        # 計算總金額
        total_value = sum(
            denom * data['total'] 
            for denom, data in breakdown.items()
        )
        
        # 計算總數量
        total_count = len(self.coins_data)
        
        return {
            'total_value': total_value,
            'total_count': total_count,
            'breakdown': breakdown
        }
    
    def reset(self):
        """重置計數器"""
        self.coins_data = []
    
    def format_summary(self) -> str:
        """
        格式化摘要文字
        
        Returns:
            摘要字串
        """
        stats = self.get_statistics()
        
        summary = f"總金額: {stats['total_value']} 元\n"
        summary += f"硬幣總數: {stats['total_count']} 個\n\n"
        
        for denom in [50, 10, 5, 1]:
            data = stats['breakdown'][denom]
            if data['total'] > 0:
                summary += f"{denom}元: {data['total']}個 "
                summary += f"(正{data['heads']}/反{data['tails']})\n"
        
        return summary


# 向後兼容：保留舊類別名稱
CoinClassifier = CoinClassifierV2


if __name__ == "__main__":
    # 測試程式碼
    classifier = CoinClassifierV2()
    counter = CoinCounter()
    
    # 模擬測試
    counter.add_coin(10, 'heads')
    counter.add_coin(10, 'tails')
    counter.add_coin(5, 'heads')
    counter.add_coin(1, 'heads')
    
    print(counter.format_summary())
