"""
Visualization of the distribution of dominant wrist information across subsets.
"""

import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New\RowPerVideo_Doms.xlsx"
df = pd.read_excel(file_path)

plt.figure(figsize=(12, 6))

category_bar_chart = df.groupby('Label')['Dominant Wrist'].value_counts().unstack().plot(kind='bar', stacked=True)
category_bar_chart.set_xlabel('Label')
category_bar_chart.set_ylabel('Count')
category_bar_chart.set_title('Distribution of Dominant Wrist by Emotion')

plt.xticks(rotation=45, ha='right', fontsize=8)
# plt.savefig('category_bar_chart.png')
plt.show()
