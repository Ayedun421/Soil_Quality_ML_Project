import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sensor_data = pd.read_csv("data/sensor_data.csv")
satellite_data = pd.read_csv("data/satellite_data.csv")

data = pd.concat([sensor_data, satellite_data], axis=1)

label_encoder = LabelEncoder()
data['soil_quality'] = label_encoder.fit_transform(data['soil_quality'])

X = data.drop("soil_quality", axis=1)
y = data["soil_quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "model/soil_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("Training completed.")
