import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import joblib

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def plot_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=sorted_features)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_and_save_models(file_path, output_dir, is_fruit=True):
    df = pd.read_csv(file_path)

    # Identify label and price columns
    if is_fruit:
        label_col = 'fruit_Commodity'
        price_col = 'fruit_Price per Unit (LKR/kg)'
    else:
        label_col = 'vegitable_Commodity'
        price_col = 'vegitable_Price per Unit (LKR/kg)'

    if label_col not in df.columns or price_col not in df.columns:
        return None

    df = df.dropna(subset=['Region', label_col, price_col])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    region_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    df['Region_encoded'] = region_encoder.fit_transform(df['Region'].astype(str))
    df['Label_encoded'] = label_encoder.fit_transform(df[label_col].astype(str))

    features_cls = [
        'Region_encoded', 'Temperature', 'Rainfall (mm)', 'Humidity (%)',
        'Crop Yield Impact Score', price_col, 'Day', 'Month', 'Year'
    ]
    features_reg = [
        'Temperature', 'Rainfall (mm)', 'Humidity (%)',
        'Crop Yield Impact Score', 'Day', 'Month', 'Year',
        'Label_encoded', 'Region_encoded'
    ]

    for col in set(features_cls + features_reg):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=features_cls + features_reg, inplace=True)

    if len(df) < 50 or df['Label_encoded'].nunique() < 2:
        return None

    # === CLASSIFICATION MODEL ===
    X_cls = df[features_cls]
    y_cls = df['Label_encoded']
    scaler_cls = StandardScaler()
    X_cls_scaled = scaler_cls.fit_transform(X_cls)

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls_scaled, y_cls, test_size=0.2, stratify=y_cls, random_state=42
    )

    model_cls = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=10,
                              use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_cls.fit(X_train_cls, y_train_cls)
    y_pred_cls = model_cls.predict(X_test_cls)
    acc = round(accuracy_score(y_test_cls, y_pred_cls) * 100, 2)

    # === REGRESSION MODEL ===
    X_reg = df[features_reg]
    y_reg = df[price_col]
    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(X_reg)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_scaled, y_reg, test_size=0.2, random_state=42
    )

    model_reg = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42)
    model_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = model_reg.predict(X_test_reg)
    r2 = round(r2_score(y_test_reg, y_pred_reg), 2)
    mae = round(mean_absolute_error(y_test_reg, y_pred_reg), 2)

    # === SAVE ALL ARTIFACTS ===
    os.makedirs(output_dir, exist_ok=True)

    # Classification
    joblib.dump(model_cls, os.path.join(output_dir, 'model_cls.pkl'))
    joblib.dump(scaler_cls, os.path.join(output_dir, 'scaler_cls.pkl'))

    # Regression
    joblib.dump(model_reg, os.path.join(output_dir, 'model_reg.pkl'))
    joblib.dump(scaler_reg, os.path.join(output_dir, 'scaler_reg.pkl'))

    # Shared encoders
    joblib.dump(region_encoder, os.path.join(output_dir, 'region_encoder.pkl'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

    # Plots
    plot_confusion_matrix(y_test_cls, y_pred_cls, label_encoder.classes_,
                          os.path.join(output_dir, 'confusion_matrix.png'))
    plot_feature_importance(model_cls, features_cls,
                            os.path.join(output_dir, 'feature_importance_cls.png'))
    plot_feature_importance(model_reg, features_reg,
                            os.path.join(output_dir, 'feature_importance_reg.png'))

    return acc, r2, mae

def process_folder(base_dir, is_fruit=True):
    subfolder = "Fruits" if is_fruit else "Vegetables"
    dataset_path = os.path.join(base_dir, subfolder)
    model_base = os.path.join("models", subfolder)
    summary_path = f"summary_{subfolder.lower()}.txt"

    os.makedirs(model_base, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        for file_name in os.listdir(dataset_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(dataset_path, file_name)
                region_name = file_name.replace(".csv", "")
                model_dir = os.path.join(model_base, region_name)

                result = train_and_save_models(file_path, model_dir, is_fruit)
                if result:
                    acc, r2, mae = result
                    summary = (f"{file_name}: "
                               f"✅ Classification Accuracy: {acc}%, "
                               f"📈 R2: {r2}, 📉 MAE: {mae} LKR")
                    print(summary)
                else:
                    summary = f"{file_name}: ⚠️ Skipped (insufficient or invalid data)"
                    print(summary)
                summary_file.write(summary + "\n")

# === Run for all Fruits and Vegetables ===
DATASET_DIR = "D:\\SEMESTER 6\\final year project\\New folder (12)\\New folder (12)\\JW New\\final\\datasets"
process_folder(DATASET_DIR, is_fruit=True)
process_folder(DATASET_DIR, is_fruit=False)