"""
In our process of feature normalization, the goal is to bring the features to a common reference point,
known as the baseline. This baseline is represented by the "Neutral" label.
For each subject, we calculate the average of all sentences' features corresponding to instances labeled
as "Neutral". Next, we perform normalization on each sample of that subject by subtracting the computed average
and dividing the result by the baseline average.
"""

import pandas as pd

excel_file = r'C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New\RowPerVideo_Doms.xlsx'
df = pd.read_excel(excel_file)


# df2
df2 = df.copy()

category_mapping = {
    'CODA sign': 0,
    'CODA speech': 1,
    'deaf': 2,
    'hearing': 3
}
df2['Category'] = df2['Category'].replace(category_mapping)

neutral_rows = df2[df2['Label'] == 'neutral']
neutral_rows = neutral_rows.drop(['Video Name', 'Video Number', 'Label', 'Dominant Wrist'], axis=1)

avg_neutral = neutral_rows.groupby(['Category', 'Person Number']).mean()
grouped = neutral_rows.groupby(['Category', 'Person Number'])
# Group by ('Category', 'Person Number') and calculate the average for each feature
avg_neutral = neutral_rows.groupby(['Category', 'Person Number']).mean()
# end of df2


exclude_columns = ['Video Name', 'Video Number', 'Person Number', 'Category', 'Label', 'Dominant Wrist']
# Normalize the data for each person
for index, row in df.iterrows():
    category = row['Category']
    person_number = row['Person Number']

    if category == 'CODA sign':
        category = 0
    elif category == 'CODA speech':
        category = 1
    elif category == 'deaf':
        category = 2
    elif category == 'hearing':
        category = 3

    person_avg = avg_neutral.loc[(category, person_number)]

    for column in df.columns:
        if column not in exclude_columns:
            df.at[index, column] = (df.at[index, column] - person_avg[column]) / person_avg[column]

new_excel_file = r'C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New\NormalizedData.xlsx'
with pd.ExcelWriter(new_excel_file) as writer:
    df.to_excel(writer, sheet_name='Normalized Data', index=False)

print("Normalization complete. New Excel file created at:", new_excel_file)
