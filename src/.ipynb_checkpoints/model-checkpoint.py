import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report

def load_clean_data(filepath):
    print(f"\n[BİLGİ] Temizlenmiş veri yükleniyor: {filepath}")
    return pd.read_csv(filepath)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n[ADIM 1] Modeller Eğitiliyor...")
    
    # 1. Logistic Regression (Baseline)
    print(" -> Logistic Regression eğitiliyor...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # 2. Random Forest (Daha Karmaşık)
    print(" -> Random Forest eğitiliyor (Bu biraz zaman alabilir)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    return lr_model, rf_model, lr_pred, rf_pred

def print_metrics(model_name, y_test, y_pred):
    print(f"\n--- {model_name} Sonuçları ---")
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy (Doğruluk): {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall (Duyarlılık): {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return f1

if __name__ == "__main__":
    DATA_PATH = "datasets/diabetes_cleaned.csv"
    MODEL_PATH = "model.pkl"
    
    df = load_clean_data(DATA_PATH)
    
    # Hedef ve Özellikler
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # Train/Test Split
    print("\n[BİLGİ] Veri seti Eğitim (%80) ve Test (%20) olarak ayrılıyor.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Eğitim
    lr_model, rf_model, lr_pred, rf_pred = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Değerlendirme
    f1_lr = print_metrics("Logistic Regression", y_test, lr_pred)
    f1_rf = print_metrics("Random Forest", y_test, rf_pred)
    
    # En iyi modeli kaydet
    best_model = rf_model if f1_rf > f1_lr else lr_model
    best_name = "Random Forest" if f1_rf > f1_lr else "Logistic Regression"
    
    print(f"\n[SONUÇ] En başarılı model: {best_name}")
    print(f"[BİLGİ] Model kaydediliyor: {MODEL_PATH}")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
        
    print("[BAŞARILI] Modelleme aşaması tamamlandı!")
