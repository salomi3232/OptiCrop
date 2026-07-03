import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('Crop_recommendation.csv')
print("Shape:", df.shape)
print(df.head())

# ── 2. EDA plots ──────────────────────────────────────────────────────────────
os.makedirs('static/plots', exist_ok=True)

# Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.drop('label', axis=1).corr(), annot=True, fmt='.2f', cmap='YlGn')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/plots/correlation_heatmap.png')
plt.close()

# Crop distribution
plt.figure(figsize=(14, 5))
df['label'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Crop Distribution in Dataset')
plt.xlabel('Crop')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('static/plots/crop_distribution.png')
plt.close()

# Feature boxplots
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    axes[i].boxplot(df[feat])
    axes[i].set_title(feat)
axes[-1].axis('off')
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.savefig('static/plots/feature_boxplots.png')
plt.close()

# NPK avg per crop (bar)
npk_avg = df.groupby('label')[['N', 'P', 'K']].mean()
npk_avg.plot(kind='bar', figsize=(14, 5), colormap='Set2', edgecolor='black')
plt.title('Average NPK Values per Crop')
plt.xlabel('Crop')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('static/plots/npk_per_crop.png')
plt.close()

# Pairplot (sample)
sample_list = [df[df['label'] == c].sample(min(10, (df['label'] == c).sum())) for c in df['label'].unique()]
sample = pd.concat(sample_list).reset_index(drop=True)
pair_fig = sns.pairplot(sample, hue='label', vars=['N', 'P', 'K', 'temperature'],
                        plot_kws={'alpha': 0.6}, height=2.2)
pair_fig.fig.suptitle('Pairplot of Key Features', y=1.02)
pair_fig.savefig('static/plots/pairplot.png')
plt.close()

print("EDA plots saved.")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
le = LabelEncoder()
df['crop_encoded'] = le.fit_transform(df['label'])

X = df[features].values
y = df['crop_encoded'].values

# Remove outliers using z-score
z = np.abs(zscore(X))
mask = (z < 3).all(axis=1)
X, y = X[mask], y[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ── 4. Train Random Forest ────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cv_scores = cross_val_score(rf, X_scaled, y, cv=5)
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. Confusion matrix plot ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix  (Accuracy: {acc:.2%})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('static/plots/confusion_matrix.png')
plt.close()

# Feature importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='teal', edgecolor='black')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('static/plots/feature_importance.png')
plt.close()

# ── 6. Save model artefacts ───────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(rf,     'model/crop_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le,     'model/label_encoder.pkl')

# Save crop stats for suitability check
crop_stats = df.groupby('label')[features].agg(['mean', 'std']).round(2)
crop_stats.to_csv('model/crop_stats.csv')

print("\nModel and artefacts saved to model/")
print("All plots saved to static/plots/")
