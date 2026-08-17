import joblib

label_encoder = joblib.load("models/label_encoder.pkl")

print(label_encoder.classes_)