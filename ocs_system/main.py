"""
OCS System - Main Application
硬幣辨識系統主程式 (靜態圖片版本)
"""

import cv2
import sys
import os
from pathlib import Path

# 加入專案路徑
sys.path.append(str(Path(__file__).parent))

from core.image_processor import ImageProcessor
from core.coin_classifier import CoinClassifier, CoinCounter


class OCSSystem:
    """OCS 硬幣辨識系統"""
    
    def __init__(self):
        """初始化系統"""
        self.processor = ImageProcessor()
        self.classifier = CoinClassifier()
        self.counter = CoinCounter()
        
        print("🪙 OCS 硬幣辨識系統已啟動")
        print("=" * 50)
    
    def process_image(self, image_path: str) -> dict:
        """
        處理單張圖片
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            辨識結果
        """
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 無法讀取圖片: {image_path}")
            return None
        
        print(f"📷 處理圖片: {image_path}")
        print(f"   尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 重置計數器
        self.counter.reset()
        
        # 檢測硬幣
        print("🔍 檢測硬幣中...")
        coins = self.processor.detect_coins_hybrid(image)
        print(f"   找到 {len(coins)} 個候選硬幣")
        
        # 分類每個硬幣
        print("🎯 分類硬幣中...")
        results = []
        
        for i, coin in enumerate(coins):
            # 提取 ROI
            roi = self.processor.extract_coin_roi(
                image, coin['x'], coin['y'], coin['radius']
            )
            
            # 提取顏色特徵
            color_features = self.processor.extract_color_features(roi)
            
            # 分類硬幣
            classification = self.classifier.classify_coin(
                roi, coin['radius'], color_features
            )
            
            # 記錄結果
            self.counter.add_coin(
                classification['denomination'],
                classification['side']
            )
            
            # 儲存完整資訊
            result = {
                'id': i + 1,
                'x': coin['x'],
                'y': coin['y'],
                'radius': coin['radius'],
                'denomination': classification['denomination'],
                'side': classification['side'],
                'confidence': classification['confidence']
            }
            results.append(result)
            
            print(f"   硬幣 #{i+1}: {classification['denomination']}元 "
                  f"({classification['side']}) - "
                  f"信心度: {classification['confidence']:.2f}")
        
        # 獲取統計資料
        stats = self.counter.get_statistics()
        
        # 繪製結果
        result_image = self._draw_results(image, results)
        
        return {
            'results': results,
            'statistics': stats,
            'result_image': result_image,
            'original_image': image
        }
    
    def _draw_results(self, image, results):
        """繪製辨識結果"""
        result_img = image.copy()
        
        for coin in results:
            x, y, radius = coin['x'], coin['y'], coin['radius']
            denom = coin['denomination']
            side = coin['side']
            
            # 根據面額選擇顏色
            colors = {1: (255, 0, 0), 5: (0, 255, 255), 
                     10: (0, 165, 255), 50: (0, 255, 0)}
            color = colors.get(denom, (255, 255, 255))
            
            # 繪製圓形
            cv2.circle(result_img, (x, y), radius, color, 3)
            
            # 標註資訊
            label = f"{denom}$ {side[0].upper()}"
            cv2.putText(result_img, label, (x - 30, y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_img
    
    def display_results(self, result_data):
        """顯示結果"""
        if result_data is None:
            return
        
        stats = result_data['statistics']
        
        print("\n" + "=" * 50)
        print("📊 辨識結果摘要")
        print("=" * 50)
        print(self.counter.format_summary())
        
        # 顯示圖片
        result_img = result_data['result_image']
        
        # 調整顯示大小
        height, width = result_img.shape[:2]
        max_width = 1200
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_img = cv2.resize(result_img, (new_width, new_height))
        
        cv2.imshow('OCS - 辨識結果', result_img)
        print("\n💡 按任意鍵關閉視窗...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_result(self, result_data, output_path: str):
        """儲存結果圖片"""
        if result_data is None:
            return
        
        cv2.imwrite(output_path, result_data['result_image'])
        print(f"💾 結果已儲存至: {output_path}")


def main():
    """主程式"""
    # 建立系統
    system = OCSSystem()
    
    # 測試圖片路徑
    test_image = "../DAY2/20251211_14_42_18_Pro.jpg"
    
    # 檢查檔案是否存在
    if not os.path.exists(test_image):
        print(f"❌ 找不到測試圖片: {test_image}")
        print("請將圖片放置於正確位置或修改路徑")
        return
    
    # 處理圖片
    result = system.process_image(test_image)
    
    # 顯示結果
    system.display_results(result)
    
    # 儲存結果
    output_path = "assets/result_output.jpg"
    system.save_result(result, output_path)
    
    print("\n✅ 處理完成！")


if __name__ == "__main__":
    main()
