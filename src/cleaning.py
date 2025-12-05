import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """
    Belirtilen dosya yolundan veri setini yükler.

    Args:
        filepath (str): CSV dosyasının yolu.

    Returns:
        pd.DataFrame: Yüklenen veri seti.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
        
    print(f"[INFO] Veri seti yükleniyor: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Boyut: {df.shape}")
    return df

def check_missing_values(df):
    """
    Veri setindeki eksik (NaN) değerleri analiz eder.

    Args:
        df (pd.DataFrame): Analiz edilecek veri seti.

    Returns:
        pd.DataFrame: Değişiklik yapılmadan döndürülen veri frame'i.
    """
    print("\n[INFO] Eksik Veri Analizi Başlatılıyor...")
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
            
    if missing_counts.empty:
        print("[INFO] Eksik veri bulunamadı.")
    else:
        print(f"[WARNING] Eksik değerler tespit edildi:\n{missing_counts}")
    
    return df

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Belirtilen kolonlar için Z-Score yöntemiyle aykırı değerleri temizler.
    
    Args:
        df (pd.DataFrame): Veri seti.
        columns (list): İşlem yapılacak kolon isimleri.
        threshold (int): Aykırı değer eşiği (Varsayılan: 3 std sapma).

    Returns:
        pd.DataFrame: Aykırı değerlerden arındırılmış veri seti.
    """
    print(f"\n[INFO] Outlier Temizliği (Z-Score > {threshold}) uygulanıyor...")
    
    df_clean = df.copy()
    initial_shape = df_clean.shape
    
    # Vektörel Z-Score hesaplaması (Daha optimize)
    for col in columns:
        mean_val = df_clean[col].mean()
        std_val = df_clean[col].std()
        z_scores = (df_clean[col] - mean_val) / std_val
        df_clean = df_clean[abs(z_scores) <= threshold]
        
    removed_count = initial_shape[0] - df_clean.shape[0]
    print(f"[INFO] Silinen satır sayısı: {removed_count}")
    print(f"[INFO] Yeni veri seti boyutu: {df_clean.shape}")
    
    return df_clean

def min_max_scaling(df, columns):
    """
    Belirtilen kolonlara Min-Max Normalizasyonu (0-1 arası ölçekleme) uygular.

    Args:
        df (pd.DataFrame): Veri seti.
        columns (list): Ölçeklenecek kolonlar.

    Returns:
        pd.DataFrame: Ölçeklenmiş veri seti.
    """
    print("\n[INFO] Min-Max Scaling başlatılıyor...")
    df_scaled = df.copy()
    
    # Scikit-learn MinMaxScaler yerine manual implementasyon (Proje gereksinimi)
    for col in columns:
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        
        if max_val - min_val == 0:
            print(f"[WARNING] {col} kolonunda varyans yok, atlanıyor.")
            continue
            
        df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
        
    print("[INFO] Scaling tamamlandı.")
    return df_scaled

if __name__ == "__main__":
    DATA_PATH = "datasets/diabetes_binary_health_indicators_BRFSS2015.csv"
    OUTPUT_PATH = "datasets/diabetes_cleaned.csv"
    
    if os.path.exists(DATA_PATH):
        # 1. Pipeline Yürütme
        df = load_data(DATA_PATH)
        df = check_missing_values(df)
        
        # 2. Preprocessing
        continuous_cols = ['BMI', 'MentHlth', 'PhysHlth'] 
        df = remove_outliers_zscore(df, continuous_cols)
        df = min_max_scaling(df, continuous_cols)
        
        # 3. Export
        print(f"\n[INFO] Temiz veriler kaydediliyor: {OUTPUT_PATH}")
        df.to_csv(OUTPUT_PATH, index=False)
        print("[SUCCESS] Veri temizleme pipeline'ı başarıyla tamamlandı.")
        
    else:
        print(f"[ERROR] Veri seti bulunamadı: {DATA_PATH}")
