import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ==========================================
# PHASE 1: Data Acquisition & Validation
# ==========================================
print("Phase 1: Loading raw Kaggle text datasets...")
train_path = "data/train.csv"
test_path = "data/test.csv"

# Guardrail: Check for local files before loading memory arrays
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError(
        "Critical Error: Missing data files in data/ subdirectory. "
        "Please download train.csv and test.csv from Kaggle."
    )

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"-> Successfully loaded {len(df_train)} training rows.")
print(f"-> Successfully loaded {len(df_test)} test evaluation rows.")

# ==========================================
# PHASE 2: NLP Text Processing & Feature Engineering
# ==========================================
print("\nPhase 2: Tokenizing and engineering TF-IDF text matrices...")

# Clean empty missing text values safely
df_train['text'] = df_train['text'].fillna("")
df_test['text'] = df_test['text'].fillna("")

# Convert text sequences into mathematical numerical vector weights
# strip_accents & stop_words remove common uninformative filler words (e.g., 'the', 'is')
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))

X_train_full = vectorizer.fit_transform(df_train['text'])
X_test_submission = vectorizer.transform(df_test['text'])
y_train_full = df_train['target'].values

# ==========================================
# PHASE 3: Defensive Cross-Validation Setup
# ==========================================
print("\nPhase 3: Splitting data defensively for local validation checks...")

# Carve out 20% of the training data as a blind safety net test to check for model overfitting
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"-> Local Training Split Matrix: {X_train.shape}")
print(f"-> Local Validation Blind Matrix: {X_val.shape}")

# ==========================================
# PHASE 4: Model Training & Throttling Parameters
# ==========================================
print("\nPhase 4: Training optimization classifier model...")
start_time = time.time()

# Implement standard regularized linear engine (C=1.0 balances model simplicity and accuracy)
model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

duration = time.time() - start_time
print(f"-> Optimization computation complete. Convergence duration: {duration:.4f} seconds.")

# ==========================================
# PHASE 5: Model Evaluation & Metric Breakdown
# ==========================================
print("\nPhase 5: Calculating validation performance matrices...")

# Evaluate performance on our blind validation set
val_predictions = model.predict(X_val)

# Core Kaggle Metric evaluation: F1-Score calculation
local_f1 = f1_score(y_val, val_predictions)
print(f"\n================ LOCAL METRICS ================")
print(f"Target Kaggle Metric (F1-Score): {local_f1:.4f}")
print("===============================================")
print("\nDetailed Classification Matrix Profiles:")
print(classification_report(y_val, val_predictions, target_names=["Non-Disaster (0)", "Disaster (1)"]))

# ==========================================
# PHASE 6: Submission Compilation & Export
# ==========================================
print("\nPhase 6: Compiling final predictions into Kaggle submission layout...")

# Extract predictions for the unlabelled test data
test_predictions = model.predict(X_test_submission)

# Map raw arrays into a clean Pandas dataframe explicitly matching Kaggle's required file format
submission_df = pd.DataFrame({
    "id": df_test["id"],
    "target": test_predictions
})

output_filename = "submission.csv"
submission_df.to_csv(output_filename, index=False)

print(f"\nSuccess! Production submission file compiled and saved locally as '{output_filename}'.")
print("You are ready to upload this file to the Kaggle Leaderboard page!")
