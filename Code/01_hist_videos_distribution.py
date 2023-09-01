import os
import matplotlib.pyplot as plt

# videos dir
path = r"C:\Users\USER\Downloads\EMOTION PRODUCTION Task 02"

# num of videos for each emotion within each category
hearing_counts = {'angry': 0, 'happy': 0, 'neutral': 0, 'sad': 0}
deaf_counts = {'angry': 0, 'happy': 0, 'neutral': 0, 'sad': 0}
coda_sign_counts = {'angry': 0, 'happy': 0, 'neutral': 0, 'sad': 0}
coda_speech_counts = {'angry': 0, 'happy': 0, 'neutral': 0, 'sad': 0}

# update the counts
for file in os.listdir(path):
    if file.endswith(".mp4"):
        file_info = file.split("_")
        emotion = file_info[2].rsplit(".", 1)[0]  # Split from the right end and remove file extension
        if "hearing" in file:
            hearing_counts[emotion] += 1
        elif "D" in file:
            deaf_counts[emotion] += 1
        elif "C" in file and "sign" in file:
            coda_sign_counts[emotion] += 1
        elif "C" in file and "speech" in file:
            coda_speech_counts[emotion] += 1

# custom colors for the bars - [red, gree, grey, blue]
colors = ['#FF0000', '#008000', '#4F4F4F', '#0000FF']

categories = ['hearing', 'deaf', 'coda_sign', 'coda_speech']
count_dicts = [hearing_counts, deaf_counts, coda_sign_counts, coda_speech_counts]

for category, counts in zip(categories, count_dicts):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(counts.keys(), counts.values(), color=colors)
    ax.set_title(category.replace('_', ' ').upper())
    ax.set_ylim([0, 45])  # y-axis limit
    ax.set_facecolor((1, 1, 1))  # background color
    ax.set_ylabel("Number of Videos")

    # Save histograms
    fig.savefig(f'webpage/webpage_files/data_distribution/{category}_plot.png', dpi=300)
    plt.show()
