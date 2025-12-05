import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64

# -----------------------------------------------------------------------------
# 1. Sayfa Konfigürasyonu
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diyabet Risk Analiz Platformu",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Premium CSS & Glassmorphism Tasarım (Ultra Modern)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Reset & Font */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #e0e0e0;
    }
    
    /* Animated Background */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar Glass */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.7);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Custom Card Style */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Headings with Gradient */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        transition: all 0.3s ease;
        font-weight: 600;
        letter-spacing: 1px;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
    }
    
    /* Widget Styling */
    .stSelectbox label, .stSlider label, .stCheckbox label, .stNumberInput label {
        color: #b0b0b0 !important;
        font-weight: 500;
    }
    
    /* Metrik Kutuları */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s;
    }
    div[data-testid="metric-container"]:hover {
        background: rgba(255,255,255,0.1);
        border-color: rgba(255,255,255,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        with open('model.pkl', 'rb') as f:
            models['lifestyle'] = pickle.load(f)
    except:
        models['lifestyle'] = None
        
    try:
        with open('model_clinical.pkl', 'rb') as f:
            models['clinical'] = pickle.load(f)
    except:
        models['clinical'] = None
    return models

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('datasets/diabetes_cleaned.csv').sample(1000)
    except:
        return None

models = load_models()
df_sample = load_sample_data()

def clean_text(text):
    """Türkçe karakterleri Latin karakterlere çevirir (PDF hatasını önlemek için)"""
    replacements = {
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text

def create_pdf(input_data, risk_score, risk_label, analysis_type="Yaşam Tarzı"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(40, 10, clean_text(f"Diyabet Risk Analiz Raporu ({analysis_type})"))
    pdf.ln(20)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, clean_text(f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"), ln=True)
    pdf.cell(0, 10, clean_text(f"Risk Durumu: {risk_label}"), ln=True)
    pdf.cell(0, 10, clean_text(f"Hesaplanan Ihtimal: %{risk_score*100:.1f}"), ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("Girdiginiz Degerler:"), ln=True)
    pdf.set_font("Arial", "", 10)
    
    for key, value in input_data.items():
        pdf.cell(0, 8, clean_text(f"{key}: {value}"), ln=True)
        
    return pdf.output(dest="S").encode("latin-1")

# -----------------------------------------------------------------------------
# 4. Sidebar - Gelişmiş Giriş Paneli
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("Hasta Profili")
    
    # Analiz Modu Seçimi
    analysis_mode = st.selectbox("Analiz Modu Seçin", ["Yaşam Tarzı Analizi", "Klinik & Genetik Analiz"])
    st.markdown("---")
    
    input_data = {}
    input_df = None
    
    if analysis_mode == "Yaşam Tarzı Analizi":
        with st.expander("📝 Temel Bilgiler", expanded=True):
            age = st.slider("Yaş Grubu", 1, 13, 9, help="1: 18-24 ... 13: 80+")
            sex = st.radio("Cinsiyet", [0, 1], format_func=lambda x: "Kadın" if x==0 else "Erkek", horizontal=True)
            
            # BMI Hesaplayıcı
            st.markdown("##### ⚖️ BMI (Vücut Kitle İndeksi)")
            bmi_manual = st.checkbox("BMI değerimi bilmiyorum, hesapla")
            if bmi_manual:
                weight = st.number_input("Kilo (kg)", 40, 200, 70)
                height = st.number_input("Boy (cm)", 140, 220, 170)
                calculated_bmi = weight / ((height/100)**2)
                st.info(f"Hesaplanan BMI: {calculated_bmi:.1f}")
                bmi = st.slider("BMI Değeri", 10.0, 50.0, float(f"{calculated_bmi:.1f}"), disabled=True)
            else:
                bmi = st.slider("BMI Değeri", 10.0, 50.0, 25.0, help="Vücut Kitle İndeksi = Kilo / Boy²")
                
            # BMI Durum Göstergesi
            if bmi < 18.5:
                st.caption("Durum: 🔵 Zayıf")
            elif 18.5 <= bmi < 25:
                st.caption("Durum: 🟢 Normal")
            elif 25 <= bmi < 30:
                st.caption("Durum: 🟠 Fazla Kilolu")
            else:
                st.caption("Durum: 🔴 Obez")
            
        with st.expander("❤️ Sağlık Geçmişi"):
            high_bp = st.checkbox("Yüksek Tansiyon", help="Doktor tarafından yüksek tansiyon teşhisi konuldu mu?")
            high_chol = st.checkbox("Yüksek Kolesterol", help="Kan kolesterol seviyeniz yüksek mi?")
            chol_check = st.checkbox("Kolesterol Kontrolü", help="Son 5 yıl içinde kolesterol ölçümü yaptırdınız mı?")
            heart_disease = st.checkbox("Kalp Hastalığı/Kriz", help="Koroner kalp hastalığı veya kalp krizi geçirdiniz mi?")
            stroke = st.checkbox("İnme (Felç)", help="Daha önce felç geçirdiniz mi?")
            
        with st.expander("🏃‍♂️ Yaşam Tarzı"):
            smoker = st.checkbox("Sigara Kullanımı", help="Hayatınız boyunca en az 5 paket (100 dal) sigara içtiniz mi?")
            phys_activity = st.checkbox("Düzenli Spor", help="Son 30 gün içinde iş dışında fiziksel aktivite veya egzersiz yaptınız mı?")
            fruits = st.checkbox("Meyve Tüketimi", help="Günde en az 1 kez meyve tüketiyor musunuz?")
            veggies = st.checkbox("Sebze Tüketimi", help="Günde en az 1 kez sebze tüketiyor musunuz?")
            hvy_alcohol = st.checkbox("Ağır Alkol Tüketimi", help="Erkekler için haftada 14, kadınlar için haftada 7 kadehden fazla alkol tüketimi.")
            diff_walk = st.checkbox("Yürüme Zorluğu", help="Yürürken veya merdiven çıkarken ciddi zorluk yaşıyor musunuz?")

        with st.expander("🏥 Genel Durum"):
            gen_hlth = st.select_slider("Genel Sağlık Algısı", options=[1, 2, 3, 4, 5], value=3, 
                                      help="1: Mükemmel, 2: Çok İyi, 3: İyi, 4: Orta, 5: Kötü")
            ment_hlth = st.number_input("Kötü Ruh Hali (Gün/Ay)", 0, 30, 0, help="Son 30 gün içinde stres, depresyon veya duygusal sorunlar yaşadığınız gün sayısı.")
            phys_hlth = st.number_input("Fiziksel Rahatsızlık (Gün/Ay)", 0, 30, 0, help="Son 30 gün içinde fiziksel hastalık veya yaralanma yaşadığınız gün sayısı.")
            any_healthcare = st.checkbox("Sağlık Sigortası", value=True, help="Herhangi bir sağlık güvenceniz var mı?")
            no_doc_cost = st.checkbox("Doktora Gidememe (Maddi)", value=False, help="Son 12 ay içinde doktora gitmeniz gerektiği halde maddi yetersizlik yüzünden gidemediğiniz oldu mu?")
            education = st.slider("Eğitim Seviyesi", 1, 6, 4, help="1: Hiç okula gitmemiş, 6: Üniversite mezunu")
            income = st.slider("Gelir Seviyesi", 1, 8, 5, help="1: <$10k ... 8: >$75k (Yıllık hane geliri)")

        # Veri Hazırlama (Yaşam Tarzı)
        input_data = {
            'HighBP': int(high_bp), 'HighChol': int(high_chol), 'CholCheck': int(chol_check), 'BMI': bmi,
            'Smoker': int(smoker), 'Stroke': int(stroke), 'HeartDiseaseorAttack': int(heart_disease),
            'PhysActivity': int(phys_activity), 'Fruits': int(fruits), 'Veggies': int(veggies),
            'HvyAlcoholConsump': int(hvy_alcohol), 'AnyHealthcare': int(any_healthcare),
            'NoDocbcCost': int(no_doc_cost), 'GenHlth': gen_hlth, 'MentHlth': ment_hlth,
            'PhysHlth': phys_hlth, 'DiffWalk': int(diff_walk), 'Sex': sex, 'Age': age,
            'Education': education, 'Income': income
        }
        input_df = pd.DataFrame(input_data, index=[0])

    else: # Klinik & Genetik Analiz
        st.info("Bu modül Pima Indians Diabetes veri seti ile eğitilmiştir.")
        with st.expander("🧬 Genetik & Klinik Veriler", expanded=True):
            pregnancies = st.number_input("Gebelik Sayısı", 0, 20, 0)
            glucose = st.number_input("Glikoz (mg/dL)", 0, 300, 120)
            blood_pressure = st.number_input("Kan Basıncı (mm Hg)", 0, 150, 70)
            skin_thickness = st.number_input("Deri Kalınlığı (mm)", 0, 100, 20)
            insulin = st.number_input("İnsülin (mu U/ml)", 0, 900, 80)
            bmi_clinical = st.number_input("BMI", 0.0, 70.0, 32.0)
            pedigree = st.number_input("Diyabet Soyağacı Fonksiyonu", 0.0, 3.0, 0.5, help="Genetik yatkınlık skoru (0.0 - 2.5 arası)")
            age_clinical = st.number_input("Yaş", 21, 100, 33)
            
        # Veri Hazırlama (Klinik)
        input_data = {
            'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi_clinical,
            'DiabetesPedigreeFunction': pedigree, 'Age': age_clinical
        }
        input_df = pd.DataFrame(input_data, index=[0])

# -----------------------------------------------------------------------------
# 5. Ana Ekran
# -----------------------------------------------------------------------------
st.markdown('<div class="glass-card"><h1>🧬 AI Destekli Diyabet Risk Analizi</h1><p>Gelişmiş Makine Öğrenmesi algoritmaları ile sağlık verilerinizi analiz edin.</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown(f'<div class="glass-card"><h3>🔍 {analysis_mode} Başlat</h3>', unsafe_allow_html=True)
    if st.button("RİSKİ HESAPLA", use_container_width=True):
        
        current_model = models['lifestyle'] if analysis_mode == "Yaşam Tarzı Analizi" else models['clinical']
        
        if current_model:
            proba = current_model.predict_proba(input_df)[0][1]
            
            # Gauge Chart (Hız Göstergesi)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diyabet Riski", 'font': {'size': 24, 'color': 'white'}},
                number = {'suffix': "%", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00d2ff"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.3)'},
                        {'range': [50, 100], 'color': 'rgba(255, 0, 0, 0.3)'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': proba * 100}}))
            
            fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Inter"})
            st.plotly_chart(fig, use_container_width=True)
            
            risk_label = "YUKSEK RISK" if proba > 0.5 else "DUSUK RISK"
            
            if proba > 0.5:
                st.markdown(f'<div style="background-color: rgba(255, 0, 0, 0.2); padding: 15px; border-radius: 10px; border: 1px solid red; text-align: center;"><h4>⚠️ YÜKSEK RİSK TESPİT EDİLDİ</h4><p>Modelimiz verilerinize dayanarak diyabet riskinizi yüksek buldu. Lütfen bir uzmana danışın.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background-color: rgba(0, 255, 0, 0.2); padding: 15px; border-radius: 10px; border: 1px solid green; text-align: center;"><h4>✅ DÜŞÜK RİSK</h4><p>Verileriniz sağlıklı bir profille eşleşiyor. İyi yaşamaya devam edin!</p></div>', unsafe_allow_html=True)
            
            # PDF İndirme Butonu
            pdf_bytes = create_pdf(input_data, proba, risk_label, analysis_mode)
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="diyabet_risk_raporu.pdf" style="text-decoration:none;"><button style="background-color:#4CAF50; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer; width:100%; margin-top:10px;">📄 Raporu İndir (PDF)</button></a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # SHAP Açıklaması (Session State'e kaydetmeye gerek yok, anlık gösterelim)
            st.session_state['show_shap'] = True
            st.session_state['input_df'] = input_df
            st.session_state['current_model'] = current_model
                
        else:
            st.error("Seçilen model yüklenemedi! Lütfen model dosyasının (pkl) mevcut olduğundan emin olun.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # SHAP Gösterimi (Her iki mod için de çalışır)
    if 'show_shap' in st.session_state and st.session_state['show_shap']:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Yapay Zeka Neden Bu Kararı Verdi?")
        st.caption("Aşağıdaki grafik, hangi özelliğin risk puanınızı ne kadar artırdığını (kırmızı) veya azalttığını (mavi) gösterir.")
        
        try:
            # SHAP Hesaplama
            model_to_explain = st.session_state['current_model']
            explainer = shap.TreeExplainer(model_to_explain)
            shap_values = explainer.shap_values(st.session_state['input_df'])
            
            # Matplotlib Dark Mode Ayarları
            plt.style.use('dark_background')
            plt.rcParams.update({
                "text.color": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "axes.edgecolor": "white"
            })
            
            # Waterfall Plot
            fig_shap, ax = plt.subplots(figsize=(10, 6))
            # Binary classification olduğu için shap_values[1] (Pozitif sınıf) kullanılır
            shap.plots.waterfall(shap.Explanation(values=shap_values[1][0], 
                                                base_values=explainer.expected_value[1], 
                                                data=st.session_state['input_df'].iloc[0],
                                                feature_names=st.session_state['input_df'].columns),
                                show=False)
            
            # Arka planı şeffaf yap
            fig_shap.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            # Yazı boyutlarını büyüt
            for text in ax.texts:
                text.set_color("white")
                text.set_fontsize(10)
                
            st.pyplot(fig_shap, transparent=True)
        except Exception as e:
            st.warning(f"SHAP grafiği oluşturulurken bir hata oluştu: {e}")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Karşılaştırmalı Analiz (Sadece Yaşam Tarzı Modunda Gösterelim, çünkü Pima için örneklem verimiz yok)
    if analysis_mode == "Yaşam Tarzı Analizi" and df_sample is not None:
        st.markdown('<div class="glass-card"><h3>📊 Karşılaştırmalı Analiz</h3>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Vücut Kitle İndeksi", "Yaş Analizi"])
        
        with tab1:
            fig_bmi = px.histogram(df_sample, x="BMI", nbins=40, title="Popülasyon BMI Dağılımı", 
                                 color_discrete_sequence=['#3a7bd5'])
            if 'BMI' in input_data:
                 fig_bmi.add_vline(x=input_data['BMI'], line_width=3, line_dash="dash", line_color="#00d2ff", annotation_text="Siz")
            
            fig_bmi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font_color="white",
                title_font_color="white",
                legend_title_font_color="white",
                xaxis=dict(showgrid=False, color="white"),
                yaxis=dict(showgrid=False, color="white")
            )
            st.plotly_chart(fig_bmi, use_container_width=True)
            
        with tab2:
            age_risk = df_sample.groupby('Age')['Diabetes_binary'].mean().reset_index()
            fig_age = px.line(age_risk, x='Age', y='Diabetes_binary', markers=True, 
                            title="Yaş İlerledikçe Risk Artışı",
                            color_discrete_sequence=['#00d2ff'])
            if 'Age' in input_data:
                fig_age.add_vline(x=input_data['Age'], line_width=3, line_dash="dash", line_color="white", annotation_text="Siz")
            
            fig_age.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font_color="white",
                title_font_color="white",
                legend_title_font_color="white",
                xaxis=dict(showgrid=False, color="white"),
                yaxis=dict(showgrid=False, color="white")
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.5);">
    <p>Geliştirilen Proje: YZ ve Makine Öğrenmesi Dersi Dönem Ödevi</p>
    <p>Ekip: Osman Yetkin, Ayberk İlcan Çirasun, Eren Aksoy</p>
</div>
""", unsafe_allow_html=True)
