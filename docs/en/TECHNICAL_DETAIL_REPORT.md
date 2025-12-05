# 📘 Technical Detail Report: Diabetes Risk Analysis Platform

This document is prepared to detail the technical infrastructure of the project, data flow, and the working logic of the algorithms used.

## 1. Project Architecture and File Structure

The project is designed in a modular structure. Each file has a specific responsibility:

*   **`src/cleaning.py` (Data Cleaning):**
    *   **Task:** Takes raw data (`datasets/diabetes_binary_health_indicators_BRFSS2015.csv`), processes it, and makes it ready for the model.
    *   **Technical:**
        *   `remove_outliers_zscore`: Detects and deletes outliers using the statistical Z-Score method.
        *   `scale_features_minmax`: Compresses data into the 0-1 range (Normalization), enabling the model to learn faster and more accurately.
    *   **Output:** `datasets/diabetes_cleaned.csv`

*   **`src/model.py` (Lifestyle Model):**
    *   **Dataset:** BRFSS 2015 (253k records).
    *   **Algorithm:** Random Forest Classifier.
    *   **Input:** Age, BMI, Smoking, Alcohol, Sports, etc.
    *   **Output:** `model.pkl`

*   **`src/model_clinical.py` (Clinical & Genetic Model):**
    *   **Dataset:** Pima Indians Diabetes (768 records).
    *   **Algorithm:** Random Forest Classifier.
    *   **Input:** Glucose, Insulin, Blood Pressure, Skin Thickness, **DiabetesPedigreeFunction (Genetic)**.
    *   **Output:** `model_clinical.pkl`

*   **`src/app.py` (Web Interface):**
    *   **Task:** It is the end-user interface that interacts with the user.
    *   **Technology:** Streamlit.
    *   **Features:**
        *   *Dual Mode Analysis:* Analysis mode selection via Sidebar.
        *   *SHAP Integration:* Provides explainability for both models.
        *   *PDF Generator:* Creates reports with Turkish character support (using `clean_text` function).

## 2. Data Flow Chart (Hybrid System)

1.  **Mode Selection:** User selects "Lifestyle" or "Clinical & Genetic" mode from Sidebar.
2.  **Input:** Relevant input fields (Form) open according to the selected mode.
3.  **Prediction:**
    *   If Lifestyle -> `model.pkl` is loaded.
    *   If Clinical -> `model_clinical.pkl` is loaded.
4.  **Result:** The `predict_proba()` function of the relevant model runs.
5.  **Explanation (SHAP):** The `TreeExplainer` of the selected model engages and explains the reason for the decision.
6.  **Reporting:** Results can be downloaded as PDF.

## 3. Newly Added Advanced Features

### A. SHAP (SHapley Additive exPlanations)
Prevents the model from being a "Black Box".
*   **How It Works?** Based on game theory. Calculates the marginal contribution of each feature (e.g., High Blood Pressure) on the result.
*   **Example:** If the result is 80% Risk; SHAP tells us that 30% of this comes from BMI, 20% from Age, 10% from Blood Pressure.

### B. PDF Reporting
*   **Library:** `fpdf`
*   **Logic:** A virtual A4 paper is created in Python code. The values entered by the user and the calculated risk result are written on this virtual paper. Then it is converted to `base64` format and made available for download via the browser.

## 4. Socioeconomic Factor Analysis (Important Findings)
As a result of user feedback and model analysis, it has been observed that **Income** and **Education** levels have a higher than expected impact on the model.

### Why? (USA Dataset Reality)
The BRFSS 2015 dataset we use is USA-sourced. In the US health system, socioeconomic status (SES) plays a decisive role on health outcomes:
1.  **Access to Health:** High-income individuals have easier access to health insurance and quality doctors.
2.  **Nutrition:** Fast Food consumption and obesity rates are statistically higher in low-income groups.
3.  **Stress:** Economic uncertainty increases chronic stress and therefore diabetes risk.

**Model's Feature Importance Ranking (Top 5):**
1.  **BMI (18.5%):** Strongest determinant.
2.  **Age (12.7%):** Risk increases as you age.
3.  **Income (10.0%):** Has a surprisingly high impact.
4.  **Physical Health (8.2%):** Person's own declaration.
5.  **Education (7.1%):** Related to health literacy.

*This situation is not a model error, but a sociological reflection of US society.*

## 5. System Requirements
Libraries required for the project to work are specified in the `requirements.txt` file.
*   `numpy < 2.0.0`: Old version is used for compatibility with SHAP library.
*   `shap == 0.44.1`: Stable explainability version.

---
*This report documents the technical depth and engineering approach of the project.*
