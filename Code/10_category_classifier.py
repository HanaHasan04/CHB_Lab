"""
An all-category Random Forest CATEGORY classifier with the evaluation metrics and feature importances.
Easy to choose any subset of categories and/or emotions.
5-fold cross validation (averaging accuracy over the 5 folds and summing the confusion matrix).
The classifier takes in 36 numeric features extracted from the video processing stage as input,
with the target variable being the category (hearing, deaf, CODA speech, or CODA sign).
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

file_path = r"C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New\NormalizedData.xlsx"
data = pd.read_excel(file_path)

# filtering the data based on emotion
selected_emotions = ['angry', 'happy', 'sad', 'neutral']
filtered_data = data[data['Label'].isin(selected_emotions)]

# filtering the data based on category
selected_categories = ['hearing', 'CODA speech', 'deaf', 'CODA sign']
filtered_data = data[data['Category'].isin(selected_categories)]

X = filtered_data.drop(['Video Name', 'Video Number', 'Person Number', 'Category', 'Label', 'Dominant Wrist'], axis=1)
y = filtered_data['Category']

# data normalization using StandardScaler, based on the training set.
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# classifier
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
confusion_mat = np.zeros((len(y.unique()), len(y.unique())))

for train_index, test_index in skf.split(X_normalized, y):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    confusion_mat += confusion_matrix(y_test, y_pred)

# Calculate average metrics
accuracy_avg = np.mean(accuracy_scores)
precision_avg = np.mean(precision_scores)
recall_avg = np.mean(recall_scores)
f1_avg = np.mean(f1_scores)

# Print average metrics
print("Accuracy: {:.2f}".format(accuracy_avg))
print("Precision: {:.2f}".format(precision_avg))
print("Recall: {:.2f}".format(recall_avg))
print("F1 Score: {:.2f}".format(f1_avg))
print("Confusion Matrix (Sum of all 5 folds):")
print(confusion_mat)

# Feature Importance
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = X.columns[sorted_indices[:4]]  # Select top 4 features

plt.figure(figsize=(7, 7))
plt.title('Feature Importance')
plt.bar(range(4), importances[sorted_indices[:4]], align='center', color='skyblue')
plt.xticks(range(4), feat_labels, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig(r"C:\Users\USER\Documents\UniversityProjects\Hagit_Lab\webpage\webpage_files\category_classifier\feat_{}.png".format('_'.join(selected_categories)), dpi=300)
plt.show()

labels = sorted(y.unique())
cm_df = pd.DataFrame(confusion_mat, index=labels, columns=labels)
plt.figure(figsize=(7, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('{}\nAccuracy: {:.3f}'.format(', '.join(selected_categories), accuracy_avg))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(r"C:\Users\USER\Documents\UniversityProjects\Hagit_Lab\webpage\webpage_files\category_classifier\conf_{}.png".format('_'.join(selected_categories)), dpi=300)
plt.show()
