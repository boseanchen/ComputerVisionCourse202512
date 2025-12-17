"""
測試配置檔 - 定義所有測試圖片的預期結果
"""

TEST_IMAGES = {
    "20251211_14_42_18_Pro.jpg": {
        "description": "測試圖片 1 - 10個硬幣",
        "total_value": 83,
        "total_count": 10,
        "coins": {
            10: {"count": 5, "heads": 0, "tails": 5},
            5: {"count": 2, "heads": 1, "tails": 1},
            1: {"count": 3, "heads": 1, "tails": 2}
        }
    },
    "20251211_14_39_07_Pro.jpg": {
        "description": "測試圖片 2 - 8個硬幣",
        "total_value": 43,
        "total_count": 8,
        "coins": {
            10: {"count": 3, "heads": 0, "tails": 3},
            5: {"count": 2, "heads": 1, "tails": 1},
            1: {"count": 3, "heads": 1, "tails": 2}
        }
    }
}


def get_test_config(image_name):
    """取得指定圖片的測試配置"""
    return TEST_IMAGES.get(image_name, None)


def get_all_test_images():
    """取得所有測試圖片名稱"""
    return list(TEST_IMAGES.keys())
