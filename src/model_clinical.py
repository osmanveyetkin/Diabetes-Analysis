import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_clinical_model():
    """Pima Indians Diabetes veri seti ile klinik modeli eğitir."""
    
    # Sütun isimleri veri setinde tanımlı olmadığı için manuel eklenir
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv('datasets/pima_diabetes.csv', names=columns)
    
    # Veri ön işleme: 0 değerlerinin (eksik veri) analizi bu aşamada atlanmıştır
    # Random Forest algoritması gürültülü veriye karşı dirençlidir
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    print(f"[INFO] Dataset Shape: {df.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[INFO] Training Clinical Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"[INFO] Model Başarısı - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    with open('model_clinical.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"[SUCCESS] Model kaydedildi: model_clinical.pkl")

if __name__ == "__main__":
    train_clinical_model()
