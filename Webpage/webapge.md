# Sign Language Project

## Comparing Emotional Expression Across Hearing, Deaf, and CODA Individuals

**Authors:** Hana Hasan & Michael Rodel  
**Supervisors:** Prof. Hagit Hel-Or, Rose Stamp, Svetlana Dachkovsky

### Introduction

Effective emotional expression is a fundamental component of human communication, and it exhibits intriguing variations among individuals with different hearing abilities. Cultural, cognitive, physiological factors, as well as individual differences like hearing abilities, contribute to these variations. Exploring the differences in emotional expression across individuals with diverse hearing abilities can offer valuable insights into the intricate dynamics between communication and hearing.

### Motivation

In this study, we aim to compare emotional expression across three distinct groups: hearing individuals, deaf individuals, and CODA (Children of Deaf Adults) individuals. Hearing individuals rely primarily on speech, while deaf individuals primarily use sign language for communication. CODA individuals, on the other hand, have been exposed to both speech and signing from early childhood due to their deaf parents, resulting in unique language and communication experiences.

### Data

- A controlled experiment was conducted to acquire data.
- The data consists of short videos of hearing people talking, deaf people signing, and CODA people doing both.
- Each video has a duration of approximately 3-6 seconds.
- Emotion labeling was done for each video, with 4 different emotions: Angry, Happy, Sad, and Neutral.

### Data Distribution

![data_dist_deaf](webpage_files/data_distribution/deaf_plot.png) ![data_dist_hearing](webpage_files/data_distribution/hearing_plot.png)  
![data_dist_coda_speech](webpage_files/data_distribution/coda_speech_plot.png) ![data_dist_coda_sign](webpage_files/data_distribution/coda_sign_plot.png)

### MediaPipe Pose: Advanced Human Pose Estimation Using Deep Learning

MediaPipe Pose is an advanced framework developed by Google that employs deep learning techniques to accurately estimate human body poses. With 33 3D landmarks, it provides a high-fidelity representation of body movements.

![mediapipe](webpage_files/mediapipe-33-keypoints.jpg)

For our study, we focused on four specific points of interest:

- The right wrist (Landmark 15)
- The left wrist (Landmark 16)
- The midpoint between the eyes (average of Landmarks 2 and 5)
- The midpoint between the shoulders (average of Landmarks 11 and 12)

**Visualizing Landmarks during Emotional Expression**

To gain insights into emotional expression during signing or speaking the angry sentence "You're wasting my time!" in Hebrew, we present three GIFs for each category. The first GIF shows the complete set of captured landmarks from MediaPipe Pose, along with the underlying skeleton. The second GIF focuses solely on our four points of interest, offering a closer examination. The third GIF presents the four landmarks in all frames of the video, providing a comprehensive view of their movements and patterns throughout the entire sequence.

**Deaf:**
![deaf](webpage_files/mediapipe_examples/deaf/frame_by_frame.gif) ![deaf](webpage_files/mediapipe_examples/deaf/our_points.gif) ![deaf](webpage_files/mediapipe_examples/deaf/all_frames.gif)  

**Hearing:**
![hearing](webpage_files/mediapipe_examples/hearing/frame_by_frame.gif) ![hearing](webpage_files/mediapipe_examples/hearing/our_points.gif) ![hearing](webpage_files/mediapipe_examples/hearing/all_frames.gif)  

**CODA sign:**
![coda-sign](webpage_files/mediapipe_examples/CODA_sign/frame_by_frame.gif) ![coda-sign](webpage_files/mediapipe_examples/CODA_sign/our_points.gif) ![coda-sign](webpage_files/mediapipe_examples/CODA_sign/all_frames.gif)  

**CODA speech:**
![coda-speech](webpage_files/mediapipe_examples/CODA_speech/frame_by_frame.gif) ![coda-speech](webpage_files/mediapipe_examples/CODA_speech/our_points.gif) ![coda-speech](webpage_files/mediapipe_examples/CODA_speech/all_frames.gif)

**Video Processing: A Two-Stage Approach**

We introduce a two-stage approach for video processing, designed to extract meaningful features from video frames. The first stage involves processing frame-wise information, while the second stage focuses on aggregating frame information to derive video-wise features.

**Frame-wise Processing:**

We start by identifying four key landmarks in each frame: the left wrist, right wrist, mid eyes, and mid shoulders. From these landmarks, we extract a set of 18 features per frame. These features include the x, y, and z coordinates for each landmark, as well as the Euclidean speed. The Euclidean speed is calculated by measuring the distance between the current frame's coordinates and the coordinates of the previous frame. For the initial frames without a previous frame, the Euclidean speed is set to zero. Additionally, we measure the distance of the wrists from the body plane, which is defined using the mid shoulders, left hip, and right hip landmarks (landmarks 23 and 24 in MediaPipe Pose).

_(18 FEATURES: for all landmarks - x, y, z coordinates and Euclidean speed ; only for the two wrists - distance from body plane.)_

**Aggregation of Frame Information:**

In the second stage, we aggregate the frame information to derive video-wise features. This process yields a total of 36 features. These features include the variance of the x, y, and z coordinates across all frames, as well as the largest eigenvalue of the covariance matrix. The covariance matrix represents the relationship between variables (x, y, z) across frames. Computing the largest eigenvalue helps identify the principal component, which indicates the direction of maximum variance. Additionally, we calculate the mean and variance of the speed, the total distance covered by each landmark (sum of all Euclidean distances), and the volume of the 3D movement for each landmark (volume of the convex hull formed by the landmark points). For the wrists, we also compute the mean and variance of the distance from the body plane.

To determine the distinction between the 'Dominant' and 'Non-dominant' wrists, we identified the wrist with the larger volume as the 'Dominant' wrist. Thus, we replaced the labels 'Left' and 'Right' with 'Dominant' and 'Non-dominant' accordingly. Eventually, we performed data normalization to achieve a standard deviation of 1 and a mean of zero.

_(36 FEATURES: for all landmarks - variance of x, y, z coordinates, largest eigenvalue, mean and variance of speed, volume of the convex hull, total distance covered ; only for the two wrists - mean and variance of distance from body)_

![dom1](webpage_files/dom_hand/by_category.png) ![dom2](webpage_files/dom_hand/by_emotion.png)

**Normalization to Baseline and Scaling:**

In our process of feature normalization, the goal is to bring the features to a common reference point, known as the baseline. This baseline is represented by the "Neutral" label.

**Normalization:** For each subject, we calculate the average of all sentences' features corresponding to instances labeled as "Neutral." This average, obtained by considering the neutral instances as a whole, represents the characteristic features of the baseline. Next, we perform normalization on each sample of that subject by subtracting the computed average. This process aligns the subject's features with the neutral baseline, ensuring that variations are relative to the neutral reference point.

**Scaling:** After normalization, we proceed to scale each subject's sample by dividing it by the baseline average. This step allows standardized comparisons across subjects and makes the normalization process relative to the baseline. It represents the normalized features as proportions of their deviation from the average of the "Neutral" instances.

**Emotion Classifier**

We utilize a _random forest_ classifier comprised of 100 trees. In order to emulate a balanced dataset, we adjust the weights assigned to each class in inverse proportion to their frequencies in the input data. For each execution, we select a different subset of our categories, including hearing, deaf, CODA sign, and CODA speech, as our data. To assess the performance of our model, we employ an 80-20 random split for training and testing, and utilize a 5-fold technique to evaluate the results on the hold-out test set. This technique involves averaging accuracy over the 5 folds and summing the confusion matrix. Our classifier takes in 36 numeric features extracted from the video processing stage as input, with the target variable being the emotion (anger, sadness, happiness, or neutrality).

![random_forest](webpage_files/RandomForestClassifier_Image.png)

**Results**

The confusion matrix is presented along with the accuracy metric and the four most significant features.

**A separate classifier for each category (hearing, deaf, CODA speech, CODA sign):**

![conf_hearing](webpage_files/emotion_classifier/conf_hearing.png) ![conf_CODA_speech](webpage_files/emotion_classifier/conf_CODA_speech.png) ![conf_deaf](webpage_files/emotion_classifier/conf_deaf.png) ![conf_CODA_sign](webpage_files/emotion_classifier/conf_CODA_sign.png)

![feat_hearing](webpage_files/emotion_classifier/feat_hearing.png) ![feat_CODA_speech](webpage_files/emotion_classifier/feat_CODA_speech.png) ![feat_deaf](webpage_files/emotion_classifier/feat_deaf.png) ![feat_CODA_sign](webpage_files/emotion_classifier/feat_CODA_sign.png)

**A separate classifier for signers (hearing, CODA speech) and speakers (deaf, CODA sign):**

![conf_hearing_CODA_speech](webpage_files/emotion_classifier/conf_hearing_CODA_speech.png) ![conf_deaf_CODA_sign](webpage_files/emotion_classifier/conf_deaf_CODA_sign.png)

![feat_hearing_CODA_speech](webpage_files/emotion_classifier/feat_hearing_CODA_speech.png) ![feat_deaf_CODA_sign](webpage_files/emotion_classifier/feat_deaf_CODA_sign.png)

**An all-category emotion classifier:**

![conf_all_categories](webpage_files/emotion_classifier/conf_all_categories.png) ![feat_all_categories](webpage_files/emotion_classifier/feat_all_categories.png)

**Category Classifier**

We utilize a _random forest_ classifier comprised of 100 trees. In order to emulate a balanced dataset, we adjust the weights assigned to each class in inverse proportion to their frequencies in the input data. For each execution, we select a different subset of our emotions as our data. To assess the performance of our model, we employ an 80-20 random split for training and testing, and utilize a 5-fold technique to evaluate the results on the hold-out test set. This technique involves averaging accuracy over the 5 folds and summing the confusion matrix. Our classifier takes in 36 numeric features extracted from the video processing stage as input, with the target variable being the category (hearing, deaf, CODA sign, and CODA speech).

![random_forest_category](webpage_files/RandomForestClassifier_Image.png)

**Results**


**Hearing vs. Deaf vs. CODA speech vs. CODA sign:<br><br>**
<img src="webpage_files\category_classifier\conf_hearing_deaf_CODA speech_CODA sign_ALL.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<img src="webpage_files\category_classifier\feat_hearing_deaf_CODA speech_CODA sign_ALL.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<br><br><br>
**Speakers vs. Signers (no CODA included):<br><br>**
<img src="webpage_files\category_classifier\conf_hearing_deaf.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<img src="webpage_files\category_classifier\feat_hearing_deaf.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<br><br><br>
**Speakers vs. Signers (no CODA included): Emotion-Wise Analysis<br><br>**
<img src="webpage_files\category_classifier\conf_hearing_deaf_angry.png" alt="data_dist_coda_sign" style="width: 270px; height: auto;">
<img src="webpage_files\category_classifier\conf_hearing_deaf_happy.png" alt="data_dist_coda_sign" style="width: 270px; height: auto;">
<img src="webpage_files\category_classifier\conf_hearing_deaf_sad.png" alt="data_dist_coda_sign" style="width: 270px; height: auto;">
<img src="webpage_files\category_classifier\conf_hearing_deaf_neutral.png" alt="data_dist_coda_sign" style="width: 270px; height: auto;">
<br>
<img src="webpage_files\category_classifier\feat_hearing_deaf_angry.png" alt="data_dist_coda_sign" style="width: 270px; height: auto;">
<img src="webpage_files\category_classifier\feat_hearing_deaf_happy.png" alt="data_dist_coda_sign" style="width: 270px; height: auto;">
<img src="webpage_files\category_classifier\feat_hearing_deaf_sad.png" alt="data_dist_coda_sign" style="width: 270px; height: auto;">
<img src="webpage_files\category_classifier\feat_hearing_deaf_neutral.png" alt="data_dist_coda_sign" style="width: 270px; height: auto;">

<br><br><br>
**Speakers vs. Signers (CODA included):<br><br>**
<img src="webpage_files\category_classifier\conf_hearing_deaf_CODA speech_CODA sign.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<img src="webpage_files\category_classifier\feat_hearing_deaf_CODA speech_CODA sign.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<br><br><br>
**Speakers vs. CODA Speakers:<br><br>**
<img src="webpage_files\category_classifier\conf_hearing_CODA speech.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<img src="webpage_files\category_classifier\feat_hearing_CODA speech.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<br><br><br>
**Deaf Signers vs. CODA Signers:<br><br>**
<img src="webpage_files\category_classifier\conf_deaf_CODA sign.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">  
<img src="webpage_files\category_classifier\feat_deaf_CODA sign.png" alt="data_dist_coda_sign" style="width: 350px; height: auto;">
<br><br><br>
<br><br><br>





**What are the distinguishing features of each emotion?**

**Hearing:**


<img src="webpage_files/binary_classifier/conf_hearing_neutral-happy.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_hearing_neutral-happy.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_hearing_neutral-happy.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
<br> 
<img src="webpage_files/binary_classifier/conf_hearing_neutral-sad.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_hearing_neutral-sad.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_hearing_neutral-sad.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
<br> 
<img src="webpage_files/binary_classifier/conf_hearing_neutral-angry.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_hearing_neutral-angry.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_hearing_neutral-angry.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">


**CODA Speech:**

<img src="webpage_files/binary_classifier/conf_CODA speech_neutral-happy.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_CODA speech_neutral-happy.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_CODA speech_neutral-happy.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
<br> 
<img src="webpage_files/binary_classifier/conf_CODA speech_neutral-sad.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_CODA speech_neutral-sad.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_CODA speech_neutral-sad.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
<br> 
<img src="webpage_files/binary_classifier/conf_CODA speech_neutral-angry.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_CODA speech_neutral-angry.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_CODA speech_neutral-angry.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">


**Deaf:**

<img src="webpage_files/binary_classifier/conf_deaf_neutral-happy.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_deaf_neutral-happy.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_deaf_neutral-happy.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
<br> 
<img src="webpage_files/binary_classifier/conf_deaf_neutral-sad.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_deaf_neutral-sad.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_deaf_neutral-sad.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
<br> 
<img src="webpage_files/binary_classifier/conf_deaf_neutral-angry.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_deaf_neutral-angry.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_deaf_neutral-angry.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">

**CODA Sign:**

<img src="webpage_files/binary_classifier/conf_CODA sign_neutral-happy.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_CODA sign_neutral-happy.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_CODA sign_neutral-happy.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
<br> 
<img src="webpage_files/binary_classifier/conf_CODA sign_neutral-sad.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_CODA sign_neutral-sad.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_CODA sign_neutral-sad.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
<br> 
<img src="webpage_files/binary_classifier/conf_CODA sign_neutral-angry.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/feat_CODA sign_neutral-angry.png" alt="data_dist_coda_sign" style="width: 250px; height: auto;">
<img src="webpage_files/binary_classifier/bestfeat_CODA sign_neutral-angry.png" alt="data_dist_coda_sign" style="width: 650px; height: 250px;">
