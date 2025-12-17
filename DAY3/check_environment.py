"""
YOLOv11 環境檢查腳本
檢查 GPU、CUDA、PyTorch 及相關套件是否正確安裝

作者: AI Course
日期: 2024
"""

import sys
import platform
import subprocess


def print_header(title: str):
    """印出標題"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_item(name: str, value: str, status: str = None):
    """印出檢查項目"""
    if status == "ok":
        icon = "[OK]"
    elif status == "warning":
        icon = "[!!]"
    elif status == "error":
        icon = "[X]"
    else:
        icon = "   "
    print(f"{icon} {name}: {value}")


def check_python():
    """檢查 Python 版本"""
    print_header("Python 環境")

    version = platform.python_version()
    print_item("Python 版本", version, "ok")
    print_item("Python 路徑", sys.executable)
    print_item("平台", platform.platform())

    # 檢查 Python 版本是否符合要求 (3.8+)
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 8:
        print_item("版本檢查", "Python 3.8+ (符合要求)", "ok")
    else:
        print_item("版本檢查", "建議使用 Python 3.8 以上版本", "warning")


def check_pytorch():
    """檢查 PyTorch 安裝"""
    print_header("PyTorch 環境")

    try:
        import torch
        print_item("PyTorch 版本", torch.__version__, "ok")
        print_item("PyTorch 路徑", torch.__file__)

        # 檢查 CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print_item("CUDA 可用", "是", "ok")
            print_item("CUDA 版本", torch.version.cuda, "ok")
            print_item("cuDNN 版本", str(torch.backends.cudnn.version()), "ok")
            print_item("cuDNN 啟用", str(torch.backends.cudnn.enabled), "ok")

            # GPU 資訊
            gpu_count = torch.cuda.device_count()
            print_item("GPU 數量", str(gpu_count), "ok")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_item(f"  GPU {i}", f"{gpu_name} ({gpu_memory:.1f} GB)")

            # 當前 GPU
            current_device = torch.cuda.current_device()
            print_item("當前使用 GPU", f"cuda:{current_device}")

            # 測試 CUDA 運算
            try:
                x = torch.rand(100, 100).cuda()
                y = torch.rand(100, 100).cuda()
                z = torch.matmul(x, y)
                print_item("CUDA 運算測試", "通過", "ok")
            except Exception as e:
                print_item("CUDA 運算測試", f"失敗: {e}", "error")
        else:
            print_item("CUDA 可用", "否 (將使用 CPU 訓練)", "warning")

        # 檢查 MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_item("MPS (Apple Silicon)", "可用", "ok")

    except ImportError:
        print_item("PyTorch", "未安裝", "error")
        print("\n建議安裝指令:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False

    return True


def check_ultralytics():
    """檢查 Ultralytics (YOLOv11) 安裝"""
    print_header("Ultralytics (YOLOv11)")

    try:
        import ultralytics
        print_item("Ultralytics 版本", ultralytics.__version__, "ok")

        # 檢查 YOLO 模型
        from ultralytics import YOLO
        print_item("YOLO 模組", "可用", "ok")

        # 檢查設定
        from ultralytics.utils import SETTINGS
        print_item("資料集目錄", SETTINGS.get('datasets_dir', 'N/A'))
        print_item("權重目錄", SETTINGS.get('weights_dir', 'N/A'))

    except ImportError:
        print_item("Ultralytics", "未安裝", "error")
        print("\n建議安裝指令:")
        print("  pip install ultralytics")
        return False

    return True


def check_opencv():
    """檢查 OpenCV 安裝"""
    print_header("OpenCV")

    try:
        import cv2
        print_item("OpenCV 版本", cv2.__version__, "ok")

        # 檢查是否支援 CUDA
        build_info = cv2.getBuildInformation()
        if "CUDA" in build_info and "YES" in build_info.split("CUDA")[1][:50]:
            print_item("OpenCV CUDA", "支援", "ok")
        else:
            print_item("OpenCV CUDA", "不支援 (使用 CPU)", "warning")

        # 檢查視訊擷取
        print_item("VideoCapture", "可用", "ok")

    except ImportError:
        print_item("OpenCV", "未安裝", "error")
        print("\n建議安裝指令:")
        print("  pip install opencv-python")
        return False

    return True


def check_other_packages():
    """檢查其他相關套件"""
    print_header("其他套件")

    packages = [
        ("numpy", "numpy"),
        ("PIL (Pillow)", "PIL"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("yaml", "yaml"),
    ]

    for name, module in packages:
        try:
            pkg = __import__(module)
            version = getattr(pkg, '__version__', 'N/A')
            print_item(name, version, "ok")
        except ImportError:
            print_item(name, "未安裝", "warning")


def check_gpu_memory():
    """檢查 GPU 記憶體使用狀況"""
    print_header("GPU 記憶體狀態")

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total - reserved

                print(f"\nGPU {i}: {props.name}")
                print(f"  總記憶體:    {total:.2f} GB")
                print(f"  已保留:      {reserved:.2f} GB")
                print(f"  已分配:      {allocated:.2f} GB")
                print(f"  可用:        {free:.2f} GB")

                # 建議 batch size
                if free > 8:
                    batch_suggestion = "batch_size=32 或更高"
                elif free > 4:
                    batch_suggestion = "batch_size=16"
                elif free > 2:
                    batch_suggestion = "batch_size=8"
                else:
                    batch_suggestion = "batch_size=4 或更低"
                print(f"  建議 batch:  {batch_suggestion}")
        else:
            print("無可用 GPU")
    except Exception as e:
        print(f"無法取得 GPU 資訊: {e}")


def run_nvidia_smi():
    """執行 nvidia-smi 顯示 GPU 狀態"""
    print_header("NVIDIA-SMI 輸出")

    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi 執行失敗")
            print(result.stderr)
    except FileNotFoundError:
        print("找不到 nvidia-smi (NVIDIA 驅動程式可能未安裝)")
    except subprocess.TimeoutExpired:
        print("nvidia-smi 執行逾時")
    except Exception as e:
        print(f"錯誤: {e}")


def test_yolo_model():
    """測試載入 YOLO 模型"""
    print_header("YOLO 模型測試")

    try:
        from ultralytics import YOLO
        import torch

        # 測試載入預訓練模型
        print("下載並載入 yolo11n.pt 測試模型...")
        model = YOLO('yolo11n.pt')
        print_item("模型載入", "成功", "ok")

        # 檢查模型資訊
        print_item("模型類型", type(model.model).__name__)

        # 測試推理
        import numpy as np
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用 {device} 進行推理測試...")

        results = model.predict(dummy_img, device=device, verbose=False)
        print_item("推理測試", "成功", "ok")

    except Exception as e:
        print_item("YOLO 測試", f"失敗: {e}", "error")


def print_summary():
    """印出總結與建議"""
    print_header("環境檢查總結")

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except:
        cuda_ok = False

    try:
        import ultralytics
        yolo_ok = True
    except:
        yolo_ok = False

    try:
        import cv2
        cv_ok = True
    except:
        cv_ok = False

    print("\n狀態:")
    print(f"  GPU/CUDA: {'可用' if cuda_ok else '不可用 (將使用 CPU)'}")
    print(f"  YOLOv11:  {'已安裝' if yolo_ok else '未安裝'}")
    print(f"  OpenCV:   {'已安裝' if cv_ok else '未安裝'}")

    if cuda_ok and yolo_ok and cv_ok:
        print("\n環境檢查通過! 可以開始訓練。")
        print("\n建議的訓練指令:")
        print("  python train_yolov11.py --model-size n --epochs 100")
    else:
        print("\n需要安裝的套件:")
        if not yolo_ok:
            print("  pip install ultralytics")
        if not cv_ok:
            print("  pip install opencv-python")
        if not cuda_ok:
            print("\n注意: 未檢測到 GPU，訓練將使用 CPU (速度較慢)")
            print("如需使用 GPU，請確認:")
            print("  1. 已安裝 NVIDIA GPU 驅動程式")
            print("  2. 已安裝 CUDA Toolkit")
            print("  3. PyTorch 為 CUDA 版本")


def main():
    """主程式"""
    print("\n" + "=" * 60)
    print("     YOLOv11 訓練環境檢查工具")
    print("=" * 60)

    check_python()
    pytorch_ok = check_pytorch()
    check_ultralytics()
    check_opencv()
    check_other_packages()

    if pytorch_ok:
        check_gpu_memory()
        run_nvidia_smi()

    # 詢問是否測試模型
    print("\n" + "-" * 60)
    try:
        response = input("是否進行 YOLO 模型載入測試? (y/n): ").strip().lower()
        if response == 'y':
            test_yolo_model()
    except:
        pass

    print_summary()

    print("\n" + "=" * 60)
    print("檢查完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
