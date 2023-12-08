@echo off

:: Create a virtual environment
python -m venv venv

:: Ensure scripts are executable (not required on Windows)
:: chmod +x Single_Domain_Prediction.sh
:: chmod +x requirments.txt

:: Activate the virtual environment
call venv\Scripts\activate

:: Install dependencies
:: pip install -r results/requirements.txt
pip install -r requirments.txt

:: To print the model accuracy test results
python Model_Accuracy_Results.py

:: Run your Python script
python SingleDomainPrediction1.py

:: Deactivate the virtual environment
deactivate

:: Delete the virtual environment
rmdir /s /q venv
