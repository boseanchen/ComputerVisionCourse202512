"""
診斷腳本 - 分析圖片並找出最佳檢測參數
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# 設定 Windows 控制台編碼 (解決 emoji 顯示問題)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 加入專案路徑
sys.path.append(str(Path(__file__).parent))


def diagnose_image(image_path):
    """診斷圖片並測試不同參數"""
    print("=" * 60)
    print("🔬 圖片診斷與參數調整")
    print("=" * 60)
    
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 無法讀取圖片: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"\n圖片資訊:")
    print(f"  路徑: {image_path}")
    print(f"  尺寸: {w}x{h}")
    print(f"  預期硬幣數量: 10 個")
    
    # 轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 測試不同的參數組合
    print("\n" + "=" * 60)
    print("測試不同的 HoughCircles 參數")
    print("=" * 60)
    
    test_configs = [
        {"name": "預設參數", "param2": 30, "minR": 15, "maxR": 100, "minDist": 30},
        {"name": "降低閾值", "param2": 20, "minR": 20, "maxR": 80, "minDist": 40},
        {"name": "更嚴格", "param2": 40, "minR": 25, "maxR": 70, "minDist": 50},
        {"name": "寬鬆範圍", "param2": 25, "minR": 15, "maxR": 90, "minDist": 35},
        {"name": "優化版", "param2": 22, "minR": 30, "maxR": 75, "minDist": 45},
    ]
    
    best_config = None
    best_count = 0
    best_diff = float('inf')
    
    for config in test_configs:
        # 預處理
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=config['minDist'],
            param1=50,
            param2=config['param2'],
            minRadius=config['minR'],
            maxRadius=config['maxR']
        )
        
        count = 0 if circles is None else len(circles[0])
        diff = abs(count - 10)
        
        status = "✅" if count == 10 else "❌"
        print(f"\n{status} [{config['name']}]")
        print(f"   param2={config['param2']}, minR={config['minR']}, "
              f"maxR={config['maxR']}, minDist={config['minDist']}")
        print(f"   檢測到: {count} 個硬幣 (差距: {diff})")
        
        if circles is not None and len(circles[0]) > 0:
            radii = [int(c[2]) for c in circles[0]]
            print(f"   半徑範圍: {min(radii)} ~ {max(radii)} px")
        
        # 記錄最佳結果
        if diff < best_diff:
            best_diff = diff
            best_count = count
            best_config = config
    
    # 顯示最佳配置
    print("\n" + "=" * 60)
    print("📊 最佳配置")
    print("=" * 60)
    print(f"\n配置名稱: {best_config['name']}")
    print(f"檢測數量: {best_count} 個 (目標: 10 個)")
    print(f"\n建議參數:")
    print(f"  param2 = {best_config['param2']}")
    print(f"  minRadius = {best_config['minR']}")
    print(f"  maxRadius = {best_config['maxR']}")
    print(f"  minDist = {best_config['minDist']}")
    
    # 使用最佳配置繪製結果
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=best_config['minDist'],
        param1=50,
        param2=best_config['param2'],
        minRadius=best_config['minR'],
        maxRadius=best_config['maxR']
    )
    
    if circles is not None:
        result_image = image.copy()
        circles = np.uint16(np.around(circles))
        
        for i, circle in enumerate(circles[0, :], 1):
            x, y, r = circle
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 3)
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(result_image, f"#{i}", (x - 10, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 儲存結果
        output_path = "diagnosis_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"\n💾 診斷結果已儲存: {output_path}")
    
    # 建議
    print("\n" + "=" * 60)
    print("💡 下一步建議")
    print("=" * 60)
    
    if best_count == 10:
        print("\n✅ 找到最佳參數！")
        print("請更新以下檔案的預設值:")
        print("  1. ui/main_window.py (GUI 預設參數)")
        print("  2. core/image_processor.py (檢測方法)")
    elif best_count < 10:
        print("\n⚠️ 檢測數量不足")
        print("建議:")
        print("  - 降低 param2 (更靈敏)")
        print("  - 降低 minRadius (檢測更小的圓)")
        print("  - 增加 maxRadius (檢測更大的圓)")
    else:
        print("\n⚠️ 檢測過多（可能有誤判）")
        print("建議:")
        print("  - 提高 param2 (更嚴格)")
        print("  - 提高 minDist (增加圓之間的距離)")
        print("  - 調整 minRadius/maxRadius 範圍")


if __name__ == "__main__":
    # 測試圖片路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(script_dir, "assets", "test_images", "20251211_14_42_18_Pro.jpg")
    
    if not os.path.exists(test_image):
        print(f"❌ 找不到測試圖片: {test_image}")
        exit(1)
    
    diagnose_image(test_image)
    
    print("\n" + "=" * 60)
    print("診斷完成！")
    print("=" * 60)
