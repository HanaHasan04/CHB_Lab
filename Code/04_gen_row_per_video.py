"""
Aggregation of frame information.
creates an Excel that consolidates information from all videos (one row per video).
36 FEATURES: for all landmarks - variance of x, y, z coordinates, largest eigenvalue,
mean and variance of speed, volume of the convex hull, total distance covered ;
only for the two wrists - mean and variance of distance from body.
"""

import os
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd


def process_excel_files(input_folder, output_folder):
    new_columns = ['Video Name', 'Video Number', 'Person Number', 'Category', 'Label',

                   'Mid Eye X Var', 'Mid Eye Y Var', 'Mid Eye Z Var', 'Mid Eye Largest Eigenvalue',
                   'Mid Eye Speed Mean', 'Mid Eye Speed Var', 'Mid Eye Volume', 'Mid Eye Distance Covered',

                   'Mid Clav X Var', 'Mid Clav Y Var', 'Mid Clav Z Var', 'Mid Clav Largest Eigenvalue',
                   'Mid Clav Speed Mean', 'Mid Clav Speed Var', 'Mid Clav Volume', 'Mid Clav Distance Covered',

                   'Right Wrist X Var', 'Right Wrist Y Var', 'Right Wrist Z Var', 'Right Wrist Largest Eigenvalue',
                   'Right Wrist Speed Mean', 'Right Wrist Speed Var', 'Right Wrist Dist Mean', 'Right Wrist Dist Var',
                   'Right Wrist Volume', 'Right Wrist Distance Covered',

                   'Left Wrist X Var', 'Left Wrist Y Var', 'Left Wrist Z Var', 'Left Wrist Largest Eigenvalue',
                   'Left Wrist Speed Mean', 'Left Wrist Speed Var', 'Left Wrist Dist Mean', 'Left Wrist Dist Var',
                   'Left Wrist Volume', 'Left Wrist Distance Covered']

    all_data = pd.DataFrame(columns=new_columns)

    for file in os.listdir(input_folder):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(input_folder, file))

            abs_columns = ['Mid Eye Speed', 'Mid Clav Speed', 'Right Wrist Speed', 'Left Wrist Speed']
            df[abs_columns] = np.abs(df[abs_columns])

            video_name = df.at[0, 'Video Name']

            print(video_name)
            video_number = df.at[0, 'Video Number']
            person_number = df.at[0, 'Person Number']
            category = df.at[0, 'Category']
            label = df.at[0, 'Label']

            # Mean of columns.
            mean_columns = ['Mid Eye Speed',
                            'Mid Clav Speed',
                            'Right Wrist Speed', 'Right Wrist Dist',
                            'Left Wrist Speed', 'Left Wrist Dist']
            values = df[mean_columns].mean()

            # Variance of columns.
            var_columns = ['Mid Eye X', 'Mid Eye Y', 'Mid Eye Z',
                           'Mid Clav X', 'Mid Clav Y', 'Mid Clav Z',
                           'Right Wrist X', 'Right Wrist Y', 'Right Wrist Z',
                           'Left Wrist X', 'Left Wrist Y', 'Left Wrist Z',
                           'Mid Eye Speed', 'Mid Clav Speed',
                           'Right Wrist Speed', 'Right Wrist Dist', 'Left Wrist Speed', 'Left Wrist Dist']
            var_values = df[var_columns].var()


            for landmark in ['Mid Eye', 'Mid Clav', 'Right Wrist', 'Left Wrist']:
                # Largest Eigenvalue.
                x = df[f'{landmark} X'].values
                y = df[f'{landmark} Y'].values
                z = df[f'{landmark} Z'].values
                landmark_data = np.column_stack((x, y, z))
                cov_matrix = np.cov(landmark_data, rowvar=False)
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                largest_eigenvalue = np.max(eigenvalues)
                values[f'{landmark} Largest Eigenvalue'] = largest_eigenvalue

                # Volume.
                points = np.array(list(zip(x, y, z)))
                volume = ConvexHull(points).volume
                values[f'{landmark} Volume'] = volume

                # Distance Covered.
                distance_sum = 0.0
                num_points = len(x)
                for i in range(num_points):
                    for j in range(i + 1, num_points):
                        dx = abs(x[i] - x[j])
                        dy = abs(y[i] - y[j])
                        dz = abs(z[i] - z[j])
                        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                        distance_sum += distance
                values[f'{landmark} Distance Covered'] = distance_sum

            # New Excel: row per video.
            all_data = all_data._append(pd.Series([
                video_name, video_number, person_number, category, label,

                var_values['Mid Eye X'], var_values['Mid Eye Y'], var_values['Mid Eye Z'], values['Mid Eye Largest Eigenvalue'],
                values['Mid Eye Speed'], var_values['Mid Eye Speed'],
                values['Mid Eye Volume'], values['Mid Eye Distance Covered'],

                var_values['Mid Clav X'], var_values['Mid Clav Y'], var_values['Mid Clav Z'], values['Mid Clav Largest Eigenvalue'],
                values['Mid Clav Speed'], var_values['Mid Clav Speed'],
                values['Mid Clav Volume'], values['Mid Clav Distance Covered'],

                var_values['Right Wrist X'], var_values['Right Wrist Y'], var_values['Right Wrist Z'],
                values['Right Wrist Largest Eigenvalue'], values['Right Wrist Speed'], var_values['Right Wrist Speed'],
                values['Right Wrist Dist'], var_values['Right Wrist Dist'],
                values['Right Wrist Volume'], values['Right Wrist Distance Covered'],

                var_values['Left Wrist X'], var_values['Left Wrist Y'], var_values['Left Wrist Z'],
                values['Left Wrist Largest Eigenvalue'], values['Left Wrist Speed'], var_values['Left Wrist Speed'],
                values['Left Wrist Dist'], var_values['Left Wrist Dist'],
                values['Left Wrist Volume'], values['Left Wrist Distance Covered']],

                index=new_columns), ignore_index=True)

    output_file = os.path.join(output_folder, 'RowPerVideo.xlsx')
    all_data.to_excel(output_file, index=False)
    print(f'Successfully processed all files and saved as {output_file}')


input_folder_ = r'C:\Users\USER\Downloads\HBC_all_vids\RowPerFrame'
output_folder_ = r'C:\Users\USER\Downloads\HBC_all_vids\RowPerVideo_New'
process_excel_files(input_folder_, output_folder_)
