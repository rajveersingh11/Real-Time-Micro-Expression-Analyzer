UNIFIED_LABELS = [
    "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"
]

def map_label(dataset_name, raw_label):
    """
    Maps raw dataset labels to a unified label space:
    anger, disgust, fear, happiness, sadness, surprise, neutral
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "fer2013":
        # FER2013 raw labels (often 0-6 or folder names)
        mapping = {
            "angry": "anger",
            "disgust": "disgust",
            "fear": "fear",
            "happy": "happiness",
            "sad": "sadness",
            "surprise": "surprise",
            "neutral": "neutral",
            # If numeric (as in CSV)
            "0": "anger",
            "1": "disgust",
            "2": "fear",
            "3": "happiness",
            "4": "sadness",
            "5": "surprise",
            "6": "neutral"
        }
        return mapping.get(str(raw_label).lower())

    elif dataset_name == "ckplus":
        # CK+ mapping: 0:neutral, 1:anger, 2:contempt, 3:disgust, 4:fear, 5:happy, 6:sadness, 7:surprise
        mapping = {
            "0": "neutral",
            "1": "anger",
            "2": None, # contempt (ignored)
            "3": "disgust",
            "4": "fear",
            "5": "happiness",
            "6": "sadness",
            "7": "surprise"
        }
        return mapping.get(str(raw_label))

    elif dataset_name == "raf-db":
        # RAF-DB mapping: 1:surprise, 2:fear, 3:disgust, 4:happiness, 5:sadness, 6:anger, 7:neutral
        mapping = {
            "1": "surprise",
            "2": "fear",
            "3": "disgust",
            "4": "happiness",
            "5": "sadness",
            "6": "anger",
            "7": "neutral"
        }
        return mapping.get(str(raw_label))

    return None
