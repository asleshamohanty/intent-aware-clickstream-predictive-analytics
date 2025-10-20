import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import pickle

DATA_PATH = "data/processed/synthetic_clickstream_final.csv"
MODEL_PATH = "models/conversion_model_xgb_enhanced.pkl"

def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(data_path)

    # --- Step 1: Funnel one-hot encoding ---
    funnel_ohe = pd.get_dummies(df['max_funnel_stage'], prefix='funnel')
    df = pd.concat([df, funnel_ohe], axis=1)

    # --- Step 2: Session depth ---
    FUNNEL = ["Landing", "Product", "Cart", "Checkout", "Purchase"]
    df['session_depth'] = df['max_funnel_stage'].apply(lambda x: FUNNEL.index(x)+1 if x in FUNNEL else 0)

    # --- Step 3: TF-IDF dimensionality reduction ---
    tfidf_cols = [c for c in df.columns if c.startswith('tfidf_')]
    if tfidf_cols:
        tfidf_matrix = df[tfidf_cols].values
        svd = TruncatedSVD(n_components=min(20, tfidf_matrix.shape[1]-1), random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        tfidf_df = pd.DataFrame(tfidf_reduced, columns=[f"tfidf_svd_{i}" for i in range(tfidf_reduced.shape[1])])
        df = pd.concat([df.drop(columns=tfidf_cols), tfidf_df], axis=1)

    # --- Step 4: Prepare features and label ---
    drop_cols = ['SessionID', 'Converted', 'max_funnel_stage', 'Keywords']
    X = df.drop(columns=drop_cols)
    y = df['Converted']

    # --- Step 5: Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Step 6: Handle class imbalance ---
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    # --- Step 7: Train XGBoost ---
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)

    # --- Step 8: Evaluate ---
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("XGBoost Enhanced Model Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # --- Step 9: Save model ---
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"XGBoost enhanced model saved to {model_path}")

    return clf

if __name__ == "__main__":
    train_model()
