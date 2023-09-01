"""
Adds dominant wrist information to the row-per-video Excel.
Dominance is determined via the volume of each of the wrists (larger volume -> dominant).
"""

import pandas as pd

# original row-per-video Excel
file_path = r"C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New\RowPerVideo.xlsx"
data = pd.read_excel(file_path)

# determine the dominant wrist based on volume
left_wrist_volume = data['Left Wrist Volume']
right_wrist_volume = data['Right Wrist Volume']
dominant_wrist = ['Left' if left_vol > right_vol else 'Right'
                  for left_vol, right_vol in zip(left_wrist_volume, right_wrist_volume)]

# add a new column for the dominant wrist
data['Dominant Wrist'] = dominant_wrist

# rename columns based on dominance
data.rename(columns=lambda col: col.replace('Left Wrist', 'Dominant Wrist')
                                  .replace('Right Wrist', 'Non-Dominant Wrist'), inplace=True)

new_file_path = r"C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New\RowPerVideo_Doms.xlsx"
data.to_excel(new_file_path, index=False)
