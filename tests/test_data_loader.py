from src.data_loader import CLASS_NAMES, get_label


def test_get_label_fresh_and_rotten():
    assert get_label("freshapples") == "fresh"
    assert get_label("RottenBanana") == "rotten"


def test_get_label_unknown_returns_none():
    assert get_label("unripe_mango") is None


def test_class_names_order_is_stable():
    # API đọc thứ tự này từ models/class_names.json, không được tự ý đổi
    assert CLASS_NAMES == ["fresh", "rotten"]
