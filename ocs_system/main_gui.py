"""
OCS System - GUI Launcher
啟動 CustomTkinter 圖形介面版本 V2 (改進版)
"""

import sys
from pathlib import Path

# 設定 Windows 控制台編碼 (解決 emoji 顯示問題)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 加入專案路徑
sys.path.append(str(Path(__file__).parent))

from ui.main_window import OCSMainWindowV2


def main():
    """啟動 GUI"""
    print("=" * 50)
    print("🪙 OCS 硬幣辨識系統 V2 - 改進版 GUI")
    print("=" * 50)
    print("正在啟動圖形介面...")
    print()
    
    try:
        app = OCSMainWindowV2()
        app.mainloop()
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        print("\n請確認已安裝所需套件:")
        print("  pip install customtkinter opencv-python pillow numpy")
        input("\n按 Enter 鍵退出...")


if __name__ == "__main__":
    main()
