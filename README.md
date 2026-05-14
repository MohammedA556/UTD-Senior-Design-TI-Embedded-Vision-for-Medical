# Surgical Instrument Tracking with Edge AI
UTD Senior Design I&II, Project #2289, TI Embedded Vision for Medical Team 2
Fall 2025 to Spring 2026

## Project Description
Design and deploy a real-time surgical instrument tracking system on the TI AM62A Edge-AI platofrm that detects, classifies, and counts surgical tools from overhead video, automating end-of-procedure count checks and reducing the risk of retained surgical items in operating rooms, low-resource hosptials, and field surgeries. 

## Team Members
Darras Rahman
Mohammed Ahmed
Renjit Joseph
Neil Prakuzhy
Young Hyun Yoon
Ramis Ahmed
Philip Cung

## Additional Credit
Credit to our Texas Instruments mentors, Andrew Shutzberg and Qutaiba Saleh, and our UTD faculty mentor, Dr. Neal Skinner. 

## Additional Libraries
The base demo is built on the TI [edgeai-gst-apps](https://github.com/TexasInstruments/edgeai-gst-apps/tree/464f70b2e780bbaed8ab048b4b548f54b75b1661) foundation. All used code is subject to Texas Instrument's disclaimer included in their respective files. 

## How to run the Demo
The demo is contained in med-gst-apps/ directory, including demo code and model. The extracted model is contained in med-gst-apps/models/tiny-lite-m3. Otherwise the tarball is included in the same models directory as tiny_lite_m3.tar.gz. 

Keep in mind the model and demo is specifically set for up to 2 gauze, 1 curved hemostat, and a scalpel at this current stage. Blue cloth background was used in training the model as well. 

Clone the repo to the AM62A Board (running Edge AI Linux SDK 10.1), and in the cloned repo run:
```
cd med-gst-apps/
./apps_python/app_edgeai.py ./configs/med_imx219.yaml
```
Alternatively you can also use python3

```
python3 apps_python/app_edgeai.py configs/med_imx219.yaml
```

## Dataset
Our tiny-lite-m3 model was trained on the following dataset, where we produced our own data and augmented it. The dataset is available below on Roboflow. 

<a href="https://universe.roboflow.com/mohammeda523/sd_augmented">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>
