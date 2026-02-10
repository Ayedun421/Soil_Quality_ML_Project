import joblib
import numpy as np

model = joblib.load("model/soil_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")

sample = np.array([[6.3,29,27,42,26,33,0.68,297]])
scaled = scaler.transform(sample)
pred = model.predict(scaled)

print("Predicted Soil Quality:", encoder.inverse_transform(pred)[0])
