import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_clinical_model():
    print("[BİLGİ] Pima Indians Diabetes Veri Seti Yükleniyor...")
    # Veri setinde başlık yok, sütun isimlerini biz veriyoruz
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv('datasets/pima_diabetes.csv', names=columns)
    
    # 0 olan değerler (Glikoz, Tansiyon vb.) eksik veri olabilir, ama şimdilik basit tutuyoruz.
    # Random Forest bu tür gürültülere karşı dayanıklıdır.
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    print(f"[BİLGİ] Veri Seti Boyutu: {df.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[BİLGİ] Klinik Model (Random Forest) Eğitiliyor...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"[SONUÇ] Model Başarısı - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    with open('model_clinical.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("[BAŞARILI] 'model_clinical.pkl' kaydedildi.")

if __name__ == "__main__":
    train_clinical_model()
