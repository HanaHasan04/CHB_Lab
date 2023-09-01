"""
Same as the Random Forest classifier, but it combines classes and ends up with 2 classes.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

file_path = r"C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New\RowPerVideo_Doms.xlsx"
data = pd.read_excel(file_path)

# Combine classes "Sad" and "Neutral" into one class, and "Happy" and "Angry" into another class
data['Label'] = data['Label'].replace({'sad': 'Sad+Neutral', 'neutral': 'Sad+Neutral',
                                       'happy': 'Happy+Angry', 'angry': 'Happy+Angry'})

# filtering the data based on category
selected_categories = ['CODA sign', 'CODA speech', 'hearing', 'deaf']
filtered_data = data[data['Category'].isin(selected_categories)]

X = filtered_data.drop(['Video Name', 'Video Number', 'Person Number', 'Category', 'Label', 'Dominant Wrist'], axis=1)
y = filtered_data['Label']

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
feat_labels = X.columns[sorted_indices]

plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), feat_labels, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

labels = sorted(y.unique())
cm_df = pd.DataFrame(confusion_mat, index=labels, columns=labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix - {}'.format(', '.join(selected_categories)))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
