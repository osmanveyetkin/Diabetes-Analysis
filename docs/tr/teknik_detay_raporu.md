# 📘 Teknik Detay Raporu: Diyabet Risk Analiz Platformu

**Bu belge**, projenin teknik altyapısını, veri akışını ve kullanılan algoritmaların çalışma mantığını detaylandırmak amacıyla hazırlanmıştır.

## 1. Proje Mimarisi ve Dosya Yapısı

Proje, modüler bir yapıda tasarlanmıştır. Her dosyanın belirli bir sorumluluğu vardır:

*   **`src/cleaning.py` (Veri Temizliği):**
    *   **Görevi:** Ham veriyi (`datasets/diabetes_binary_health_indicators_BRFSS2015.csv`) alır, işler ve modele hazır hale getirir.
    *   **Teknik:**
        *   `remove_outliers_zscore`: İstatistiksel Z-Skoru yöntemiyle aykırı değerleri (Outlier) tespit eder ve siler.
        *   `scale_features_minmax`: Verileri 0-1 aralığına sıkıştırarak (Normalization) modelin daha hızlı ve doğru öğrenmesini sağlar.
    *   **Çıktı:** `datasets/diabetes_cleaned.csv`

*   **`src/model.py` (Yaşam Tarzı Modeli):**
    *   **Veri Seti:** BRFSS 2015 (253k kayıt).
    *   **Algoritma:** Random Forest Classifier.
    *   **Girdi:** Yaş, BMI, Sigara, Alkol, Spor vb.
    *   **Çıktı:** `model.pkl`

*   **`src/model_clinical.py` (Klinik & Genetik Model):**
    *   **Veri Seti:** Pima Indians Diabetes (768 kayıt).
    *   **Algoritma:** Random Forest Classifier.
    *   **Girdi:** Glikoz, İnsülin, Tansiyon, Deri Kalınlığı, **DiabetesPedigreeFunction (Genetik)**.
    *   **Çıktı:** `model_clinical.pkl`

*   **`src/app.py` (Web Arayüzü):**
    *   **Görevi:** Kullanıcı ile etkileşime giren son kullanıcı arayüzüdür.
    *   **Teknoloji:** Streamlit.
    *   **Özellikler:**
        *   *Çift Modlu Analiz:* Sidebar üzerinden analiz modu seçimi.
        *   *SHAP Entegrasyonu:* Her iki model için de açıklanabilirlik sağlar.
        *   *PDF Generator:* Türkçe karakter destekli rapor oluşturur (`clean_text` fonksiyonu ile).

## 2. Veri Akış Şeması (Hibrit Sistem)

1.  **Mod Seçimi:** Kullanıcı Sidebar'dan "Yaşam Tarzı" veya "Klinik & Genetik" modunu seçer.
2.  **Giriş:** Seçilen moda göre ilgili input alanları (Form) açılır.
3.  **Tahmin:**
    *   Eğer Yaşam Tarzı -> `model.pkl` yüklenir.
    *   Eğer Klinik -> `model_clinical.pkl` yüklenir.
4.  **Sonuç:** İlgili modelin `predict_proba()` fonksiyonu çalışır.
5.  **Açıklama (SHAP):** Seçilen modelin `TreeExplainer`'ı devreye girer ve kararın nedenini açıklar.
6.  **Raporlama:** Sonuçlar PDF olarak indirilebilir.

## 3. Yeni Eklenen Gelişmiş Özellikler

### A. SHAP (SHapley Additive exPlanations)
Modelin bir "Kara Kutu" olmasını engeller.
*   **Nasıl Çalışır?** Oyun teorisine dayanır. Her bir özelliğin (örneğin Yüksek Tansiyon), sonuç üzerindeki marjinal katkısını hesaplar.
*   **Örnek:** Eğer sonuç %80 Risk ise; SHAP bize bunun %30'unun BMI'dan, %20'sinin Yaştan, %10'unun Tansiyondan geldiğini söyler.

### B. PDF Raporlama
*   **Kütüphane:** `fpdf`
*   **Mantık:** Python kodunda sanal bir A4 kağıdı oluşturulur. Kullanıcının girdiği değerler ve hesaplanan risk sonucu bu sanal kağıda yazılır. Ardından `base64` formatına çevrilerek tarayıcı üzerinden indirilmesi sağlanır.

## 4. Sosyoekonomik Faktör Analizi (Önemli Bulgular)
Kullanıcı geri bildirimleri ve model analizleri sonucunda, **Gelir (Income)** ve **Eğitim (Education)** seviyelerinin model üzerinde beklenenden yüksek bir etkisi olduğu gözlemlenmiştir.

### Neden? (ABD Veri Seti Gerçeği)
Kullandığımız BRFSS 2015 veri seti ABD kaynaklıdır. ABD sağlık sisteminde sosyoekonomik statü (SES), sağlık çıktıları üzerinde belirleyici bir rol oynar:
1.  **Sağlığa Erişim:** Yüksek gelirli bireylerin sağlık sigortasına ve kaliteli doktora erişimi daha kolaydır.
2.  **Beslenme:** Düşük gelir gruplarında "Fast Food" tüketimi ve obezite oranı istatistiksel olarak daha yüksektir.
3.  **Stres:** Ekonomik belirsizlik, kronik stresi ve dolayısıyla diyabet riskini artırır.

**Modelin Özellik Önem Sıralaması (Top 5):**
1.  **BMI (%18.5):** En güçlü belirleyici.
2.  **Yaş (%12.7):** Yaşlandıkça risk artar.
3.  **Gelir (%10.0):** Şaşırtıcı derecede yüksek bir etkiye sahip.
4.  **Fiziksel Sağlık (%8.2):** Kişinin kendi beyanı.
5.  **Eğitim (%7.1):** Sağlık okuryazarlığı ile ilişkilidir.

*Bu durum modelin hatası değil, ABD toplumunun sosyolojik bir yansımasıdır.*

## 5. Sistem Gereksinimleri
Projenin çalışması için gerekli kütüphaneler `requirements.txt` dosyasında belirtilmiştir.
*   `numpy < 2.0.0`: SHAP kütüphanesi ile uyumluluk için eski versiyon kullanılmıştır.
*   `shap == 0.44.1`: Stabil açıklanabilirlik sürümü.

---
*Bu rapor, projenin teknik derinliğini ve mühendislik yaklaşımını belgelemektedir.*
