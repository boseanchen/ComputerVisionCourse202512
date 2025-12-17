"""
測試腳本 - 驗證硬幣辨識準確度
使用 20251211_14_42_18_Pro.jpg 作為測試樣本
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# 設定 Windows 控制台編碼 (解決 emoji 顯示問題)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 加入專案路徑
sys.path.append(str(Path(__file__).parent))

from core.image_processor import ImageProcessor
from core.coin_classifier import CoinClassifier, CoinCounter


def analyze_test_image(image_path):
    """分析測試圖片"""
    print("=" * 60)
    print("🔬 硬幣辨識測試分析")
    print("=" * 60)
    print(f"\n測試圖片: {image_path}")
    
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 無法讀取圖片")
        return
    
    print(f"圖片尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 初始化
    processor = ImageProcessor()
    classifier = CoinClassifier()
    counter = CoinCounter()
    
    # === 測試不同的檢測方法 ===
    print("\n" + "=" * 60)
    print("📊 測試不同檢測方法")
    print("=" * 60)
    
    # 方法 1: Contour Detection
    print("\n[方法 1] Contour Detection")
    coins_contour = processor.detect_coins_contours(image)
    print(f"  檢測到: {len(coins_contour)} 個硬幣")
    
    # 方法 2: HoughCircles
    print("\n[方法 2] HoughCircles")
    coins_hough = processor.detect_coins_hough(image)
    print(f"  檢測到: {len(coins_hough)} 個硬幣")
    
    # 方法 3: Hybrid
    print("\n[方法 3] Hybrid (混合)")
    coins_hybrid = processor.detect_coins_hybrid(image)
    print(f"  檢測到: {len(coins_hybrid)} 個硬幣")
    
    # === 分析最佳方法的結果 ===
    print("\n" + "=" * 60)
    print("🎯 使用 Contour Detection 進行詳細分析")
    print("=" * 60)
    
    coins = coins_contour
    
    # 分析每個硬幣
    print(f"\n檢測到的硬幣資訊:")
    print(f"{'ID':<4} {'半徑':<8} {'面積':<10} {'圓形度':<10} {'顏色':<10}")
    print("-" * 50)
    
    for i, coin in enumerate(coins, 1):
        # 提取 ROI
        roi = processor.extract_coin_roi(image, coin['x'], coin['y'], coin['radius'])
        
        # 顏色特徵
        color_features = processor.extract_color_features(roi)
        color_type = "金色" if color_features['is_golden'] else "銀色"
        
        print(f"{i:<4} {coin['radius']:<8} {coin.get('area', 0):<10.0f} "
              f"{coin.get('circularity', 0):<10.3f} {color_type:<10}")
    
    # === 分類硬幣 ===
    print("\n" + "=" * 60)
    print("💰 硬幣分類結果")
    print("=" * 60)
    
    counter.reset()
    results = []
    
    for i, coin in enumerate(coins, 1):
        roi = processor.extract_coin_roi(image, coin['x'], coin['y'], coin['radius'])
        color_features = processor.extract_color_features(roi)
        classification = classifier.classify_coin(roi, coin['radius'], color_features)
        
        counter.add_coin(classification['denomination'], classification['side'])
        
        results.append({
            'id': i,
            'radius': coin['radius'],
            'denomination': classification['denomination'],
            'side': classification['side'],
            'confidence': classification['confidence']
        })
        
        print(f"硬幣 #{i}: {classification['denomination']}元 "
              f"({classification['side']}) - 半徑: {coin['radius']}")
    
    # === 統計結果 ===
    print("\n" + "=" * 60)
    print("📈 統計結果")
    print("=" * 60)
    
    stats = counter.get_statistics()
    
    print(f"\n總金額: {stats['total_value']} 元")
    print(f"硬幣總數: {stats['total_count']} 個\n")
    
    for denom in [50, 10, 5, 1]:
        data = stats['breakdown'][denom]
        if data['total'] > 0:
            print(f"【{denom}元】: {data['total']} 個 "
                  f"(正{data['heads']}/反{data['tails']}) "
                  f"= {denom * data['total']} 元")
    
    # === 驗證結果 ===
    print("\n" + "=" * 60)
    print("✅ 結果驗證")
    print("=" * 60)
    
    expected = {
        'total_value': 83,
        'total_count': 10,
        10: 5,
        5: 2,
        1: 3
    }
    
    print(f"\n預期結果:")
    print(f"  總金額: {expected['total_value']} 元")
    print(f"  硬幣總數: {expected['total_count']} 個")
    print(f"  10元: {expected[10]} 個")
    print(f"  5元: {expected[5]} 個")
    print(f"  1元: {expected[1]} 個")
    
    print(f"\n實際結果:")
    print(f"  總金額: {stats['total_value']} 元 ", end="")
    if stats['total_value'] == expected['total_value']:
        print("✅")
    else:
        print(f"❌ (差距: {stats['total_value'] - expected['total_value']})")
    
    print(f"  硬幣總數: {stats['total_count']} 個 ", end="")
    if stats['total_count'] == expected['total_count']:
        print("✅")
    else:
        print(f"❌ (差距: {stats['total_count'] - expected['total_count']})")
    
    for denom in [10, 5, 1]:
        actual = stats['breakdown'][denom]['total']
        print(f"  {denom}元: {actual} 個 ", end="")
        if actual == expected[denom]:
            print("✅")
        else:
            print(f"❌ (差距: {actual - expected[denom]})")
    
    # === 建議調整 ===
    print("\n" + "=" * 60)
    print("💡 參數調整建議")
    print("=" * 60)
    
    if stats['total_count'] < expected['total_count']:
        print("\n⚠️ 檢測到的硬幣數量不足")
        print("建議調整:")
        print("  1. 降低 HoughCircles 的 param2 (當前: 30 → 建議: 20-25)")
        print("  2. 調整最小半徑 (當前: 15 → 建議: 10-12)")
        print("  3. 增加明暗對比 (當前: 2.0 → 建議: 2.5-3.0)")
    elif stats['total_count'] > expected['total_count']:
        print("\n⚠️ 檢測到過多硬幣（可能有誤判）")
        print("建議調整:")
        print("  1. 提高 HoughCircles 的 param2 (當前: 30 → 建議: 35-40)")
        print("  2. 提高圓形度閾值 (當前: 0.7 → 建議: 0.75-0.8)")
    
    if stats['total_value'] != expected['total_value']:
        print("\n⚠️ 總金額不正確（分類錯誤）")
        print("建議調整:")
        print("  1. 檢查顏色判斷閾值")
        print("  2. 調整尺寸分類閾值")
        print("  3. 收集更多樣本進行校正")
    
    # === 儲存結果圖片 ===
    result_image = processor.draw_coins(image.copy(), coins)
    output_path = "test_result_analysis.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\n💾 結果圖片已儲存: {output_path}")
    
    return results, stats


if __name__ == "__main__":
    # 測試圖片路徑（修正為正確的相對路徑）
    import os
    
    # 獲取腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 建立正確的圖片路徑
    test_image = os.path.join(script_dir, "assets", "test_images", "20251211_14_42_18_Pro.jpg")
    
    # 檢查檔案是否存在
    if not os.path.exists(test_image):
        print(f"❌ 找不到測試圖片: {test_image}")
        print(f"\n請確認圖片位於: {script_dir}\\assets\\test_images\\")
        input("\n按 Enter 鍵退出...")
        exit(1)
    
    # 執行分析
    analyze_test_image(test_image)
    
    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)
