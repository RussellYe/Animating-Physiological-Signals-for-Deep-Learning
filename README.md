# Animating-Physiological-Signals-for-Deep-Learning

This project propose a technique to perform video classification on ["Gingerbread Animating" videos](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing/Low%20resolution%20original%20videos) using PyTorch. 

**Dataset Description**

The "Gingerbread Animation" videos dataset is created from an experiment conducted at the ANU Research School of Computer Science. [1] There are two versions of the dataset, including one low-resolution video dataset with 24 low-resolution videos (240p) and a high-resolution video (720p) dataset with only 20 high-resolution videos. There are 24 students participated in the experiment and each of them is required to listen to randomly two types of music (8 pieces of music in total) among three types of music which are classical music, instrumental music and pop music, while the students' physiological signals are recorded during the experiment. Each of the "Gingerbread Animation" video represents each student's physiological data. The low-resolution video dataset is complete and every low-resolution video represents each student's responses in the experiment. For the high-resolution video dataset, the first four students' data is missing so it only contains the data of students from No.4 student to No.24 student. This project will mainly focus on the low-resolution dataset as a proof of concept, and the results generated from the low-resolution dataset should be compared to the high-resolution dataset in the future. 

The videos in both video datasets are in RGB colour and are of the same size and format. There are four types of physiological data recorded in the videos, which are Electrodermal Activity (EDA), Blood Volume Pulse (BVP), Skin Temperature (ST) and Pupil Dilation (PD). [2] Each physiological data is recorded in different frequency, where EDA, BVP, ST and PD signals are recorded by a rate of 4Hz, 64Hz, 4Hz, and 60Hz respectively. [1] Thus, the PD and BVP signals will display in a higher frequency in the video compared to EDA and ST. 

A sample frame of the "Gingerbread Animating" video is shown in the figure below. 

![Sample frames](./Figure/sample.png)



**Project Description**

The aim of the project is to predict and classify whether a sequence of frames (like the sequence shown above) from the "Gingerbread Animation" video represents classical music, instrumental music or pop music. In this project, I have implemented a hybrid deep learning model called convolutional recurrent neural network (CRNN) to perform video classification on the "Gingerbread Animation" videos dataset. The CRNN model consists of a convolutional layer with ResNet152 or ResNet-34 pre-trained model and a recurrent layer with Long Short Term Memory (LSTM). 

The core block of code and the architecture of the CRNN model including Gingerbread_ResNetCRNN.py, ResNetCRNN_check_prediction.py, check_video_predictions.ipynb, functions.py files are implemented based on https://github.com/HHTseng/video-classification. 

**Environment Used**

Python: 3.7.3

PyTorch: 1.4.0

torchvision: 0.5.0

CUDA: 11.0

GPU: Nvidia GTX1660Ti GDDR6 

## Code Running Instructions

### 1. Download or Clone the pre-processed Gingerbread Animation video dataset.

Please initially download or clone this project with the [data_preprocessing](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing) folder. The folder [High resolution original videos](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing/High%20resolution%20original%20videos) stores the high resolution Gingerbread Animation videos (20 videos), which are not used for training in this project. The folder [Low resolution original videos](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing/Low%20resolution%20original%20videos) stores the the low resolution Gingerbread Animation videos (24 videos), which are the original dataset used for training in this project. The folder [Segmentation Info](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing/Segmentation%20Info) displays the video segmentation information. The folder [Video cuts collection](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing/Video%20cuts%20collection) stores all necessary frames extracted from the original Gingerbread Animation videos. 

### 2. Pre-process Gingerbread Animation videos data
The original Gingerbread video data is in the folder [Low resolution original videos](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing/Low%20resolution%20original%20videos). The file [Video_Segmentation.ipynb](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/blob/main/Data%20pre-processing/Video_Segmentation.ipynb) is to process the original video data and divide the videos into 8 segments for each video, and store the video segments in the folder [Video cuts collection/Video Segments/](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing/Video%20cuts%20collection/Video%20Segments). Then, the video segments stored in the folder [Video cuts collection/Video Segments/](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/tree/main/Data%20pre-processing/Video%20cuts%20collection/Video%20Segments) is processed by [Frame_extraction_all.ipynb](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/blob/main/Data%20pre-processing/Frame_extraction_all.ipynb) and the output is stored in the folder [/Video cuts collection/Video frames collection/train_all1] and [/Video cuts collection/Video frames collection/train_all2] and [/Video cuts collection/Video frames collection/train_all3]. Finally, the data stored in the folders mentioned above are processed by [Frame_extraction_final.ipynb](https://github.com/RussellYe/Animating-Physiological-Signals-for-Deep-Learning/blob/main/Data%20pre-processing/Frame_extraction_final.ipynb) which will output the dataset used for training which are [/Video cuts collection/Video frames collection/training_split1] and [/Video cuts collection/Video frames collection/training_split2] and [/Video cuts collection/Video frames collection/training_split3]. 

The 1, 2, 3 at the end of the file name indicates the frame extraction method used. For example, training_split3 means this dataset is extracted by the 3-frames-per-second data extraction method, and training_all1 means this dataset is extracted by the 1-frame-per-second data extraction method. Since these folders are too large and hard to upload in Github. Please download them in at https://drive.google.com/file/d/1Zpx9Pzs-Wlp41PR-Ua_CQXk-krFzFp1J/view?usp=sharing. Then put all the folders including train_all1, train_all2, train_all3, training_split1, training_split2 and training_split3 under the directory of Video frames collection. For example, /Video cuts collection/Video frames collection/train_all1. 

The data used for training is stored in training_split1, training_split2, training_split3 folders, which represents the frame extraction method of 1-frame-per-second, 2-frames-per-second and 3-frames-per-second respectively. 



