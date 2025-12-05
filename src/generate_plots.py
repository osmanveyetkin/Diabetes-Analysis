import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans' # Safe font

def load_data():
    return pd.read_csv("datasets/diabetes_cleaned.csv")

def generate_plots():
    print("[INFO] Loading data...")
    df = load_data()
    
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # --- 1. Model Comparison Plot ---
    models = ['Logistic Regression', 'Random Forest']
    
    # Calculate metrics
    metrics = {
        'Model': [],
        'Metric': [],
        'Value': []
    }
    
    for model_name, pred in [('Logistic Regression', lr_pred), ('Random Forest', rf_pred)]:
        metrics['Model'].extend([model_name] * 4)
        metrics['Metric'].extend(['Accuracy', 'F1-Score', 'Recall', 'Precision'])
        metrics['Value'].extend([
            accuracy_score(y_test, pred),
            f1_score(y_test, pred),
            recall_score(y_test, pred),
            precision_score(y_test, pred)
        ])
        
    df_metrics = pd.DataFrame(metrics)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics, x='Metric', y='Value', hue='Model', palette='viridis')
    plt.title('Model Performans Karşılaştırması', fontsize=14)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('datasets/model_karsilastirma.png', dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: datasets/model_karsilastirma.png")
    plt.close()

    # --- 1.5 Confusion Matrix Heatmap (Random Forest) ---
    print("[INFO] Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test, rf_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Random Forest - Confusion Matrix (Hata Matrisi)', fontsize=14)
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek Değer (Actual)')
    plt.xticks([0.5, 1.5], ['Sağlıklı', 'Diyabetli'])
    plt.yticks([0.5, 1.5], ['Sağlıklı', 'Diyabetli'])
    plt.tight_layout()
    plt.savefig('datasets/confusion_matrix.png', dpi=300)
    plt.close()
    plt.savefig('datasets/confusion_matrix.png', dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: datasets/confusion_matrix.png")
    
    # --- 2. Feature Importance Plot ---
    print("[INFO] Generating Feature Importance Plot...")
    importances = rf.feature_importances_
    feature_names = X.columns
    
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values('Importance', ascending=False).head(15) # Top 15
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='magma')
    plt.title('Diyabet Tahmininde En Önemli 15 Faktör (Random Forest)', fontsize=14)
    plt.xlabel('Önem Düzeyi')
    plt.tight_layout()
    plt.savefig('datasets/ozellik_onemi.png', dpi=300)
    plt.close()
    plt.savefig('datasets/ozellik_onemi.png', dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: datasets/ozellik_onemi.png")

if __name__ == "__main__":
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    generate_plots()
