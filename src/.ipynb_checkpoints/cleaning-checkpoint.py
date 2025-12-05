import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """
    Veri setini yükler ve temel bilgileri ekrana basar.
    Neden?: Analize başlamadan önce verinin boyutunu ve yapısını anlamamız gerekir.
    """
    print(f"\n[BİLGİ] Veri seti yükleniyor: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[BİLGİ] Veri seti boyutu: {df.shape}")
    print("\n[BİLGİ] İlk 5 satır:")
    print(df.head())
    return df

def check_missing_values(df):
    """
    Eksik verileri kontrol eder.
    Neden?: Eksik veriler modelin öğrenmesini bozar. Gerçek hayatta bu verileri doldurmamız (imputation) 
    veya silmemiz gerekir. Bu veri setinde eksik veri olup olmadığını manuel olarak kontrol ediyoruz.
    """
    print("\n[ADIM 1] Eksik Veri Analizi")
    # Hazır fonksiyon (df.isnull().sum()) yerine mantığı anlamak için:
    missing_counts = {}
    for col in df.columns:
        # Her kolondaki NaN (Not a Number) değerleri sayıyoruz
        count = 0
        for val in df[col]:
            if pd.isna(val):
                count += 1
        if count > 0:
            missing_counts[col] = count
            
    if not missing_counts:
        print("[SONUÇ] Harika! Veri setinde eksik değer bulunamadı.")
    else:
        print(f"[UYARI] Eksik değerler bulundu: {missing_counts}")
    
    return df

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Z-Score yöntemi ile aykırı değerleri (outliers) temizler.
    Neden?: Aykırı değerler (örneğin BMI'ın 100 olması), modelin genel örüntüyü öğrenmesini zorlaştırır.
    Z-Score mantığı: Bir değer ortalamadan kaç standart sapma uzakta? 
    Genellikle 3 standart sapmadan uzak veriler aykırı kabul edilir.
    """
    print(f"\n[ADIM 2] Aykırı Değer Temizliği (Z-Score > {threshold})")
    print(f"[BİLGİ] İşlem yapılacak kolonlar: {columns}")
    
    original_size = len(df)
    df_clean = df.copy()
    
    for col in columns:
        # 1. Ortalama Hesapla (Mean)
        mean_val = df_clean[col].mean()
        # 2. Standart Sapma Hesapla (Std Dev)
        std_val = df_clean[col].std()
        
        # 3. Z-Score Hesapla: z = (x - mean) / std
        # Vektörel işlem kullanarak hız kazanıyoruz ama mantık döngüyle aynıdır.
        z_scores = (df_clean[col] - mean_val) / std_val
        
        # 4. Filtreleme: Mutlak değeri threshold'dan küçük olanları tut
        df_clean = df_clean[abs(z_scores) <= threshold]
        
    removed_count = original_size - len(df_clean)
    print(f"[SONUÇ] Toplam {removed_count} satır aykırı değer olduğu için silindi.")
    print(f"[BİLGİ] Yeni veri seti boyutu: {df_clean.shape}")
    
    return df_clean

def min_max_scaling(df, columns):
    """
    Min-Max Scaling (Normalizasyon) işlemi.
    Neden?: Farklı ölçekteki verileri (Örn: Yaş 0-100, Gelir 0-10000) 0 ile 1 arasına sıkıştırarak
    modelin (özellikle KNN, Neural Networks gibi mesafe temelli modellerin) daha adil öğrenmesini sağlarız.
    Formül: (X - X_min) / (X_max - X_min)
    """
    print("\n[ADIM 3] Özellik Ölçeklendirme (Min-Max Scaling)")
    df_scaled = df.copy()
    
    for col in columns:
        # Manuel hesaplama
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        
        # Bölme işleminde sıfıra bölünme hatasını önlemek için kontrol
        if max_val - min_val == 0:
            print(f"[UYARI] {col} kolonunda tüm değerler aynı, scaling atlanıyor.")
            continue
            
        # Formülü uygula
        df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
        
    print("[SONUÇ] Seçilen kolonlar 0-1 aralığına ölçeklendi.")
    return df_scaled

if __name__ == "__main__":
    # Dosya yolları
    DATA_PATH = "datasets/diabetes_binary_health_indicators_BRFSS2015.csv"
    OUTPUT_PATH = "datasets/diabetes_cleaned.csv"
    
    # 1. Yükle
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        
        # 2. Eksik Veri Kontrolü
        df = check_missing_values(df)
        
        # 3. Aykırı Değer Temizliği
        # Sadece sürekli (continuous) değişkenlerde outlier aranır. 
        # Binary (0/1) kolonlarda outlier aranmaz.
        continuous_cols = ['BMI', 'MentHlth', 'PhysHlth'] 
        df = remove_outliers_zscore(df, continuous_cols)
        
        # 4. Scaling
        # Genellikle tüm feature'lar scale edilebilir ama binary'ler zaten 0-1'dir.
        # Yine de eğitim amaçlı sürekli değişkenleri scale edelim.
        df = min_max_scaling(df, continuous_cols)
        
        # 5. Kaydet
        print(f"\n[BİLGİ] Temizlenmiş veri kaydediliyor: {OUTPUT_PATH}")
        df.to_csv(OUTPUT_PATH, index=False)
        print("[BAŞARILI] Veri temizliği tamamlandı!")
        
    else:
        print(f"[HATA] Dosya bulunamadı: {DATA_PATH}")
