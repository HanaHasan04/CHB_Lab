"""
Visualization of the values of different features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel(r"C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New\‏‏NormalizedData_Copy.xlsx")

selected_people = ['hearing', 'deaf', 'CODA speech', 'CODA sign']
# selected_people = ['CODA sign']
data = data[data['Category'].isin(selected_people)]

selected_categories = ['angry', 'happy', 'sad', 'neutral']
# selected_categories = ['neutral', 'angry']
data = data[data['Label'].isin(selected_categories)]

# list of "best" features.
best_features = ['Mid Eye Speed Var', 'Mid Clav Speed Mean', 'Mid Clav X Var', 'Mid Eye Largest Eigenvalue']

labels = data['Label'].unique()

# initialize the scaler.
scaler = MinMaxScaler()

# colors of the emotions
emotion_colors = {'angry': 'red', 'happy': 'green', 'sad': 'blue', 'neutral': 'gray'}

fig, axes = plt.subplots(nrows=1, ncols=len(best_features), figsize=(15, 5))
for i, feature in enumerate(best_features):
    feature_data = data[feature].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(feature_data)
    # plot points for each label separately with different colors.
    for j, label in enumerate(labels):
        label_data = scaled_data[data['Label'] == label]
        axes[i].scatter(np.full(label_data.shape, j), label_data, color=emotion_colors[label], alpha=0.5, marker='o', s=10)
    axes[i].set_xticks(np.arange(len(labels)))
    axes[i].set_xticklabels(labels)
    axes[i].set_ylabel('')
    axes[i].set_xlabel(' ')
    axes[i].set_title(f'{feature}')
    axes[i].set_ylim(0, 1)
plt.tight_layout()

selected_people_filename = '-'.join(selected_people)
selected_categories_filename = '-'.join(selected_categories)
filename = "{}_{}.png".format(selected_people_filename, selected_categories_filename)
plt.savefig(r"C:\Users\USER\Documents\UniversityProjects\Hagit_Lab\webpage\webpage_files\binary_classifier\bestfeat_{}".format(filename), dpi=300)

plt.show()
