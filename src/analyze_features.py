import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance():
    print("[BİLGİ] Model yükleniyor...")
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("[HATA] 'model.pkl' bulunamadı. Lütfen önce modeli eğitin.")
        return

    # Özellik isimleri (cleaning.py veya model.py'den alınmalı, burada manuel tanımlıyoruz)
    feature_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
        'Education', 'Income'
    ]

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_imp = feature_imp.sort_values(by='Importance', ascending=False)

        print("\n[SONUÇ] Özellik Önem Düzeyleri (Feature Importance):")
        print(feature_imp)

        # Görselleştirme (Opsiyonel, terminal çıktısı yeterli olabilir ama dosyaya kaydedelim)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp)
        plt.title('Random Forest - Özellik Önem Düzeyleri')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("\n[BİLGİ] Grafik 'feature_importance.png' olarak kaydedildi.")
        
    else:
        print("[UYARI] Bu model feature_importances_ özelliğine sahip değil.")

if __name__ == "__main__":
    analyze_feature_importance()
