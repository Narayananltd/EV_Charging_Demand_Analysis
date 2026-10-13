
from pathlib import Path
import json
import joblib

root = Path(__file__).resolve().parent
models_dir = root / "models"

for filename in [
    "best_regression_model.joblib",
    "tuned_xgboost_model.joblib",
]:
    path = models_dir / filename
    print(f"\n{'=' * 55}")
    print("FILE:", filename)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    obj = joblib.load(path)
    print("Loaded object type:", type(obj).__name__)

    if isinstance(obj, dict):
        print("Dictionary keys:", list(obj.keys()))
        for key, value in obj.items():
            print(f"  {key!r}: type={type(value).__name__}")
            if hasattr(value, "n_features_in_"):
                print("    n_features_in_:", value.n_features_in_)
            if hasattr(value, "feature_names_in_"):
                print("    feature_names_in_:", list(value.feature_names_in_))
    else:
        print("Has predict():", callable(getattr(obj, "predict", None)))
        print("Feature count:", getattr(obj, "n_features_in_", "unknown"))
        print("Feature names:", getattr(obj, "feature_names_in_", "unknown"))

for filename in [
    "best_regression_metadata.json",
    "tuned_xgboost_metadata.json",
]:
    path = models_dir / filename
    print(f"\n{'=' * 55}")
    print("METADATA:", filename)
    if path.exists():
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
    else:
        print("METADATA FILE NOT FOUND")