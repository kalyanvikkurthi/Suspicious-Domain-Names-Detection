# Project Title

Our project aims to recognize and uncover suspicious domain names generated by malware through the utilization of Domain Generation Algorithms (DGAs). The key objective is to create a powerful machine learning classifier model proficient in identifying the existence of these dubious domain names. This model will scrutinize the traits and patterns linked to algorithmically generated domains, facilitating the detection of possible malicious activities.


## Installation

This project is organized into three main folders:

1. **Feature Pre-Processing:**
   - The "DataSet Generator 2Lakh.py" script, located in this folder, processes input files (DGA_Netlab360.txt and top2lakhdomains.csv) to generate two output files, namely benign2lakh.csv and dga2lakh.csv.
   - Important: Before running "DataSet Generator 2Lakh.py," make sure to execute the `requirements.txt` file in sudo mode.

2. **Model Training:**
   - The "ModelDevelopment.ipynb" file, within this folder, utilizes the "benign2lakh.csv" and "dga2lakh.csv" inputs to construct various classifier models. Specifically, it generates 15-feature models and 39-feature models separately.
   - Note: Due to the time-intensive nature of model training (building three models on approximately 4 lakh domains takes around 1-3 hours on a standard Colab Pro machine), we've pre-trained the models and stored them in the "all_models_15_featuresJ.joblib" file in the results folder. You don't need to run these files to obtain outputs, but we've included them for reference on how we trained our models.

3. **Results:**
   - This folder contains all the necessary files and packages for smooth execution during the evaluation phase.
   - To review our test phase results, execute the "model_test_results.sh" script from the terminal.
   - Additionally, the "Single_Domain_Prediction.sh" script installs all the required Python packages to run the code. Note that you don't need to run this script individually; it is triggered by "Single_Domain_Prediction.sh" for convenience.

## Code Run Sequence

Follow these steps to run the code:

1. **Install Dependencies:**
   - Open the terminal and navigate to the "Feature Pre-Processing" directory.
   - Run the following command to install dependencies:
     ```bash
     pip3 install -r requirements.txt
     ```

2. **Run "DataSet Generator 2Lakh.py":**
   - Stay in the same directory and execute the following command:
     ```bash
     python3 "DataSet Generator 2Lakh.py"
     ```
   - This will generate the files "benign2lakh.csv" and "dga2lakh.csv."

3. **Model Training:**
   - Navigate to the "Model Training" directory.
   - Open and run the "ModelDevelopment.ipynb" file, ensuring it has access to the "benign2lakh.csv" and "dga2lakh.csv" files.

Note: The provided sequence assumes that you are using Python 3. Make sure your environment is set up correctly, and all required dependencies are installed before running the code.


# How to Test the Model

To evaluate the performance of the model, follow these steps:

## 1. Navigate to Results Directory

Open your terminal and navigate to the "results" directory where the model evaluation scripts are located.

```bash
cd path/to/your/results


Note: Download the required files from Google Drive:

Download 39 Features Pickle File : https://drive.google.com/file/d/1YKAZEQ8-06UZSRCCKL8AwOVVzOR1Lws_/view?usp=drive_link 

Download 15 Features JOBlib File : https://drive.google.com/file/d/1RoqlrdjkHi7uo_yHdo-cAtb_h-z3S9iI/view?usp=drive_link

Ensure that both files are in your "results" folder

2. Run Single_Domain_Prediction.sh

Execute the provided shell script to initiate the testing process. This script triggers the model_accuracy_results.py to assess the model's accuracy using the pre-trained files for both 39 features and 15 features.


./Single_Domain_Prediction.sh


3. View Prediction
The script will utilize SingleDomainPrediction1.py to make predictions for both benign (legitimate) and DGA (non-legitimate) domains. The model will predict whether the provided domain is benign or generated by a DGA.



