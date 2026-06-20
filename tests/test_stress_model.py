from src.inference.stress_model import StressModel

def test_compute_score(stress_model):
    features = {
        "eyebrow_raise": 0.5,
        "lip_tension": 0.5,
        "blink_intensity": 0.5,
        "head_nod": 0.5,
        "symmetry_delta": 0.5
    }
    score = stress_model.compute_score(features)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_classify_stress(stress_model):
    assert stress_model.classify_stress(0.1) == "Calm"
    assert stress_model.classify_stress(0.4) == "Slight Stress"
    assert stress_model.classify_stress(0.8) == "High Stress"

def test_predict(stress_model):
    features = {
        "eyebrow_raise": 0.2,
        "lip_tension": 0.2,
        "blink_intensity": 0.2,
        "head_nod": 0.2,
        "symmetry_delta": 0.2
    }
    res = stress_model.predict(features)
    assert isinstance(res, dict)
    assert "stress_score" in res
    assert "stress_level" in res
    assert 0.0 <= res["stress_score"] <= 1.0
