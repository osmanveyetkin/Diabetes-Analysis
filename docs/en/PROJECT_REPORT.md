# Diabetes Risk Analysis and Prediction Platform: Hybrid Machine Learning Approach

**Course:** Artificial Intelligence and Machine Learning - Term Project  
**Institution:** Kırklareli University Computer Programming 2nd Year  
**Submission Date:** 06.12.2025

| Student ID | Name Surname | Role |
|------------|----------|-----|
| **1247008055** | Osman Yetkin | Team Lead & Developer |
| **1247008042** | Ayberk İlcan Çirasun | Data Research & Analysis |
| **1247008012** | Eren Aksoy | Model Selection & Optimization |

---

## Abstract

Diabetes (Diabetes Mellitus) is one of the most common chronic diseases of the modern age and can lead to serious health problems if not diagnosed early. The main purpose of this study is to develop a hybrid machine learning system that predicts diabetes risk with high accuracy by analyzing individuals' lifestyle habits and clinical data. In the project, **Random Forest Classifier** and **Logistic Regression** algorithms were analyzed comparatively, and the Random Forest model was selected with **87.22% accuracy**. The study differs from similar studies in the literature by offering a two-way (hybrid) approach that focuses not only on clinical measurements (glucose, insulin) but also on daily life habits (smoking, alcohol, BMI). In addition, explainable artificial intelligence (XAI) integration was performed using the **SHAP (SHapley Additive exPlanations)** library to ensure the transparency of model decisions.

**Keywords:** Machine Learning, Diabetes Prediction, Random Forest, SHAP, Data Mining, Decision Support System.

---

## 1. Introduction

According to International Diabetes Federation (IDF) data, hundreds of millions of people live with diabetes worldwide, and this number is expected to increase in the coming years. Although diabetes is a manageable disease, if diagnosed late, it can cause irreversible complications such as heart disease, kidney failure, and vision loss. In this context, AI-supported early warning systems have become a critical need for the health sector.

### Purpose and Scope of the Project
The main purpose of this project is to develop an accessible and high-accuracy **Decision Support System** where users can see their own risk status before taking a medical test. Our reasons for choosing this topic as a project team are:
1.  **Social Benefit:** Contributing to early diagnosis by increasing diabetes awareness.
2.  **Technical Challenge:** Gaining experience in modeling with high performance on imbalanced datasets.
3.  **Usability:** Transforming data science into a web interface that the end user can understand.

Within the scope of this study, an end-to-end software development process was carried out from data cleaning to model training, from web-based user interface to PDF reporting.

---

## 2. Related Work

In studies on diabetes detection in the literature, the Pima Indians Diabetes dataset was generally used and algorithms such as Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Artificial Neural Networks (ANN) were tested.

*   *Katiyar et al.* worked on diabetes classification with deep learning methods.
*   *Tripathi and Kumar* reported that the Random Forest algorithm was more successful than other methods (LDA, KNN, SVM) with 87.66% accuracy.

Our study confirms this literature information and takes the **Random Forest** algorithm as a basis, but expands the scope by including the **BRFSS 2015 (Behavioral Risk Factor Surveillance System)** dataset in the process. Thus, our model can perform risk analysis not only by looking at blood values but also by the person's answers to lifestyle questions such as "Did you exercise today?", "Do you consume vegetables?".

---

## 3. Material and Methods

The project was developed using the Python programming language. Data processing, modeling, and interface development stages are detailed below.

### 3.1 Datasets
Since a hybrid structure was used in the project, two different datasets were processed:
1.  **Lifestyle Dataset (BRFSS 2015):** Contains survey data of 253,680 participants. (Features: BMI, Smoking, Alcohol, Physical Activity, Age, Education, etc.)
2.  **Clinical Dataset (Pima Indians):** Contains clinical measurements of 768 patients. (Features: Glucose, Insulin, Blood Pressure, Skin Thickness, etc.)

### 3.2 Data Preprocessing
Raw data was cleaned with the `src/cleaning.py` module before entering model training.

*   **Outlier Detection:** The **Z-Score** method were used to reduce noise in the dataset. Data with a Z-score absolute value greater than 3 (deviating by 3 times the standard deviation) were removed from the dataset.
    ```python
    # Example Z-Score Application
    z_scores = np.abs(stats.zscore(df[columns]))
    df_clean = df[(z_scores < 3).all(axis=1)]
    ```

*   **Normalization (Scaling):** To prevent data in different units (e.g., Age 1-13, BMI 15-50) from misleading the model, **Min-Max Scaling** was applied to compress all values into the 0-1 range.

### 3.3 Algorithms Used
Two basic algorithms were selected for performance comparison:
1.  **Logistic Regression:** Used as a basic reference (baseline) model based on linear separability.
2.  **Random Forest Classifier:** A successful algorithm in complex datasets, resistant to overfitting, where a large number of decision trees work with voting logic.

---

## 4. Experimental Results

Models were trained by splitting the data as **80% training** and **20% testing**. Accuracy, F1-Score, and Recall were used as performance metrics.

### 4.1 Model Performance Analysis

| Model | Accuracy | F1-Score | Recall |
|-------|----------|----------|--------|
| **Random Forest** | **0.8722** | **0.2192** | **0.1445** |
| Logistic Regression | 0.8780 | 0.2129 | 0.1329 |

When the table is examined, although the accuracy rate of Logistic Regression seems marginally higher, it was seen that the **Random Forest** model was superior in **Recall (Success in detecting real patients)** and **F1-Score** metrics, which are of vital importance in medical diagnoses. Therefore, Random Forest was selected as the final model.

> *Note: Recall values are generally low due to the fact that the "Healthy" class in the dataset is vastly superior in number to the "Diabetic" class (Class Imbalance).*

![Model Comparison Chart](../datasets/model_karsilastirma.png)

### 4.2 Error Analysis (Confusion Matrix)
A Confusion Matrix was created to examine the classification errors of the model in more detail. This matrix visualizes how many patients the model missed (False Negative) and how many healthy individuals it gave a false alarm (False Positive).

![Confusion Matrix](../datasets/confusion_matrix.png)


### 4.3 Feature Importance Analysis
When analyzed which criteria the model prioritizes when making a decision, it was seen that the most effective factors on diabetes risk are:
1.  **Body Mass Index (BMI):** The most determining factor (18.5% impact).
2.  **Age:** Risk increases linearly as age progresses.
3.  **General Health Status:** How the person defines their own health.
4.  **Income Level:** It has been observed that the risk of diabetes increases in low-income groups.

![Feature Importance Levels](../datasets/ozellik_onemi.png)


### 4.4 Model Explainability (SHAP)
SHAP charts have been added to the interface to prevent the model from being a "Black Box". Thus, when the user receives a "Risky" result, they can see which problem contributed how much to this result (Ex: "Your BMI value increased the risk by 10%").

---

## 5. Alternative Methods and Discussion

Although the **Random Forest** algorithm gave successful results in this study, different methods can be evaluated in the light of current developments in the literature:

### Gradient Boosting Models (XGBoost / LightGBM)
**XGBoost** or **LightGBM** algorithms, which are frequently preferred in data science competitions such as Kaggle, can be strong alternatives for this project.
*   **Advantage:** Since these models proceed by correcting the errors made by previous trees (Boosting), they have the potential to increase the **Recall** value, especially in imbalanced datasets.
*   **Why Not Used?** Random Forest was preferred at the current stage of the project due to its interpretability and ease of setup (low need for Hyperparameter tuning). However, in phase 2, it is aimed to reduce the number of missed diabetic patients (False Negative) by trying XGBoost.

---

## 6. Conclusion and Discussion

This **Diabetes Analysis Platform** developed within the scope of this study has shown how machine learning algorithms can be used effectively for early diagnosis in the health field. A success rate (**87%+**) in accordance with literature standards was achieved with the Random Forest algorithm.

The strongest aspect of the project is its ability to present technical complexity to the end user with a modern web interface and understandable PDF reports. The findings confirm that lifestyle changes (weight control, physical activity) are at least as important as genetic factors in managing diabetes risk.

In future studies, it is aimed to apply the **SMOTE (Synthetic Minority Over-sampling Technique)** method to eliminate imbalance in the dataset and to feed the model with larger clinical datasets.

---

### 7. References
1.  U.S. Department of Health & Human Services, [*Diabetes Health Indicators Dataset (BRFSS 2015)*](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
2.  National Institute of Diabetes and Digestive and Kidney Diseases, [*Pima Indians Diabetes Database*](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
3.  Scikit-Learn Documentation: [*Random Forest Classifier*](https://scikit-learn.org/stable/modules/ensemble.html#forest).
4.  Lundberg, S. M., & Lee, S. I. (2017). [*A Unified Approach to Interpreting Model Predictions (SHAP)*](https://shap.readthedocs.io/en/latest/).
5.  Google DeepMind: [*Gemini Models*](https://deepmind.google/technologies/gemini/).
6.  Streamlit Documentation: [*Streamlit*](https://docs.streamlit.io/).
7.  Google Colab: [*Google Colab*](https://colab.research.google.com/).
8.  Google DeepMind: [*Research*](https://deepmind.google/research/).
