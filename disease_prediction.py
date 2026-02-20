import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('diabetes_cleaned.csv')
print(data.head())
print(data.info())
print(data.describe())

print("Dataset shape:",data.shape)
print(data['Outcome'].value_counts())

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    data[col]=data[col].replace(0,np.nan)

print(data.isnull().sum())

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    data[col] = data[col].fillna(data[col].mean())
print(data.isnull().sum())

data.to_csv("diabetes_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as diabetes_cleaned.csv")


plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

x = data.drop('Outcome',axis=1)
y = data['Outcome']

print(x.head())
print(x.shape)

print(y.head(25))
print(y.shape)

print(x.columns)
print(y.unique())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data, transform both
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("x_train_scaled shape:",x_train_scaled.shape)

print(x_train_scaled.mean(axis=0))


print(x_train_scaled.shape)
print(x_test_scaled.shape)

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train_scaled, y_train)

# Make predictions on test data
y_pred = log_model.predict(x_test_scaled)

print(y_pred[:10])

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=300, class_weight={0: 1, 1: 2},random_state=42)

rf_model.fit(x_train,y_train)

rf_pred = rf_model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("Logistic Regression Accuracy:", accuracy)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print(confusion_matrix(y_test, y_pred))      # Logistic
print(confusion_matrix(y_test, rf_pred))     # Random Forest

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['bloodpressure'])
    skin_thickness = float(request.form['skinthickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    proba = rf_model.predict_proba(input_scaled)[0][1]

    if proba > 0.4:
        result = f"⚠️ High Risk of Diabetes(Risk: {proba*100:.1f}%)"
    else:
        result = f"✅ Low Risk of Diabetes(Risk: {proba*100:.1f}%)"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

