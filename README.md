Adaptive accessible behavioural biometric authentication for Smartphones  


#Overview
This project file consists of two main components:

#Android Application
Real-time behavioural data capture (keystrokes + gestures)
Continuous authentication using statistical scoring
Accessibility-aware calibration and sensitivity controls


#Python Machine Learning Pipeline
Offline statistical evaluation of behavioural features
Dataset processing and feature extraction
Privacy-preserving analysis

#Project Structure
Dissertation_Submission/
├── android_code/        # Android Studio project
├── ml_pipeline_code/        # Python pipeline
├── README.txt


#Requirement:

Android
Android Studio (latest stable version)
Android SDK (API 26+)

Python
Python 3.10 or later
pip

#Running the Android App
Open Android Studio
Click Open and select the android_app folder
Allow Gradle to sync
Run on emulator pixel 6 (tirimisu) or physical device

#Running the Python Pipeline
Step 1: Install dependencies
Navigate to the pipeline folder:
cd ml_pipeline_code
pip install -r requirements.txt


#Step 2: Add datasets
Datasets are NOT included due to file size constraints.
Place them in the following directories:
ml_pipeline_code/data/raw/cmu/
ml_pipeline_code/data/raw/aalto_ite/

  Required datasets:
  CMU Keystroke Dynamics Dataset
   https://www.cs.cmu.edu/~keystroke/
   Aalto ITE Mobile Typing Dataset
   ITE Typing dataset

#Step 3: Run the pipeline
python main.py


Notes
All generated outputs can be reproduced by running the pipeline
The Android app and pipeline are conceptually linked through shared feature formats



Author
Aliya Abdullahi 001354380
BSc Computer Security and Forensics Dissertation
University of Greenwich
