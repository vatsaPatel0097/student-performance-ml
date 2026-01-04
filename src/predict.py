import pandas as pd
import joblib
from pathlib import Path


# Absolute paths (safe & portable)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


def load_model():
    """Load trained ML pipeline"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(MODEL_PATH)


def predict_math_score(input_data: dict):
    """
    Predict math score for a single student.

    input_data keys:
    - gender
    - race/ethnicity
    - parental level of education
    - lunch
    - test preparation course
    - reading score
    - writing score
    """

    model = load_model()

    # Convert dict to DataFrame
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    return float(prediction[0])


if __name__ == "__main__":
    # Example test input
    sample_input = {
        "gender": "female",
        "race/ethnicity": "group B",
        "parental level of education": "bachelor's degree",
        "lunch": "standard",
        "test preparation course": "completed",
        "reading score": 72,
        "writing score": 74
    }

    result = predict_math_score(sample_input)
    print(f"Predicted Math Score: {result:.2f}")
