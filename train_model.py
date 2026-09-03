# Train and evaluate prediction model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading feature data...")
df = pd.read_csv('matches_features.csv')
print(f"Loaded {len(df)} rows")

# Prepare features and target
feature_cols = [
    'home_form', 'away_form',
    'home_goals_avg', 'away_goals_avg',
    'home_xg_avg', 'away_xg_avg',
    'h2h_home_win_pct', 'h2h_draw_pct', 'h2h_away_win_pct'
]

X = df[feature_cols]
y = df['actual_result']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Target classes: {le.classes_}")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {len(X_train)} matches")
print(f"Test set: {len(X_test)} matches")

# Handle class imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {weight_dict}")

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight=weight_dict,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(feature_cols, model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")

# Cross-validation
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

# Save model and label encoder
joblib.dump(model, 'laliga_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(feature_cols, 'feature_columns.pkl')

print("\nSaved:")
print("  - laliga_model.pkl")
print("  - label_encoder.pkl")
print("  - feature_columns.pkl")

# Test prediction function
print("\nTesting prediction function...")
test_match = X_test.iloc[0:1]
pred = model.predict(test_match)
pred_proba = model.predict_proba(test_match)

print(f"Sample features: {test_match.iloc[0].to_dict()}")
print(f"Prediction: {le.inverse_transform(pred)[0]}")
print("Probabilities:")
for i, prob in enumerate(pred_proba[0]):
    print(f"  {le.classes_[i]}: {prob:.1%}")