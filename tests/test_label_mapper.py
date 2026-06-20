from src.preprocessing.label_mapper import map_label

def test_map_label():
    # Test FER2013
    assert map_label("fer2013", "angry") == "anger"
    assert map_label("fer2013", 0) == "anger"
    assert map_label("fer2013", "neutral") == "neutral"
    
    # Test CK+
    assert map_label("ckplus", 1) == "anger"
    assert map_label("ckplus", 2) is None  # Contempt is ignored
    assert map_label("ckplus", 7) == "surprise"
    
    # Test RAF-DB
    assert map_label("raf-db", 1) == "surprise"
    assert map_label("raf-db", 6) == "anger"
    
    # Test unknown dataset
    assert map_label("unknown_dataset", "something") is None
