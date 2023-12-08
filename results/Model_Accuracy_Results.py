from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from tabulate import tabulate as tabulate_function  

import tabulate
import warnings

# Suppress FutureWarnings related to is_sparse deprecation
warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess_data(seed=42):
    # Read the benign and dga data with low_memory=False
    benign = pd.read_csv("benign2lakh.csv")
    benign['class'] = 'benign'

    dga = pd.read_csv("dga2lakh.csv", low_memory=False)
    dga['class'] = 'dga'

    # Combine the dataframes
    data = pd.concat([benign, dga])

    # Update specific columns
    data['matched_word'] = data['matched_word'].apply(lambda x: '1' if pd.notna(x) and x != '' else '0').astype('category')
    data['feedback_warning'] = data['feedback_warning'].apply(lambda x: '1' if pd.notna(x) and x != '' else '0').astype('category')
    data['class'] = data['class'].astype('category')

    # Remove the 4th column (index 3 in zero-based indexing)
    data = data.drop(data.columns[3], axis=1)

    # Prepare features (X) and target (y)
    X = data.drop(columns=['class', data.columns[0]])  # Drop target and first column (Domain Names)
    y = data['class']

    # Handle NaN values by imputing with mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=seed)

    # Get feature names as a list
    feature_names = list(data.drop(columns=['class', data.columns[0]]).columns)

    return X_train, X_test, y_train, y_test, feature_names




#result on the 39 features datset
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

# Load the trained models
with open('all_models_39_features.pkl', 'rb') as file:
    trained_models = pickle.load(file)

seed = 42
X_train, X_test, y_train, y_test, feature_names = preprocess_data(seed)

# Select only 100 records for each set
X_train = X_train[:1000]
X_test = X_test[:1000]
y_train = y_train[:1000]
y_test = y_test[:1000]


# Dictionary to store metrics and confusion matrices for each model
model_results = {}

# Test each model on the test data and calculate metrics
for model_name, model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # TP / (TP + FN)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)

    # Save metrics and confusion matrix for each model
    model_results[model_name] = {
        'Accuracy': accuracy,
        'Kappa': kappa,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'ConfusionMatrix': cm
    }

# Convert results to a DataFrame for display
results_df = pd.DataFrame(model_results).T

# Print title
print("Results for the 39 features model\n")


# Print results including confusion matrices in a tabular format
headers = ["Model", "Accuracy", "Kappa", "Sensitivity", "Specificity", "Confusion Matrix"]
rows = []

for model_name, result in model_results.items():
    row = [
        model_name,
        f"{result['Accuracy']:.4f}",
        f"{result['Kappa']:.4f}",
        f"{result['Sensitivity']:.4f}",
        f"{result['Specificity']:.4f}",
        f"\n{result['ConfusionMatrix']}\n"
    ]
    rows.append(row)

# Print the tabulated results

# print(tabulate(rows, headers=headers, tablefmt='grid'))
# Print the tabulated results
print(tabulate_function(rows, headers=headers, tablefmt='grid'))



from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
import joblib


import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from tabulate import tabulate



def preprocess_data(seed, selected_features):
    # Read the benign and dga data
    benign = pd.read_csv("benign2lakh.csv")
    benign['class'] = 'benign'
    dga = pd.read_csv("dga2lakh.csv", low_memory=False)
    dga['class'] = 'dga'
    data = pd.concat([benign, dga])

    # Select only the specified features and the target
    data = data[selected_features + ['class']]

    y = data['class']

    data = data.drop(columns=['class'])

    # # Handle NaN values and standardize
    # imputer = SimpleImputer(strategy='mean')
    # scaler = StandardScaler()
    # X = imputer.fit_transform(data.drop(columns=['class']))
    # X = scaler.fit_transform(X)
    # y = data['class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.10, random_state=seed)
    return X_train, X_test, y_train, y_test, data

seed = 1234
selected_features = ['mean1', 'var1', 'sd1', 'sd2', 'ccc', 'cvc', 'vcc', 'vcv', 'cc', 'vv', 'nch', 'Uchar', 'cc', 'cv', 'vc']
X_train, X_test, y_train, y_test , data = preprocess_data(seed, selected_features)


# Load the trained models for 15 features using joblib
trained_models_15_features = joblib.load('all_models_15_featuresJ.joblib')

# Select only 100 records for each set
X_train = X_train[:1000]
X_test = X_test[:1000]
y_train = y_train[:1000]
y_test = y_test[:1000]

# Dictionary to store metrics and confusion matrices for each model
model_results_15_features = {}

# Test each model on the test data and calculate metrics
for model_name, model in trained_models_15_features.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # TP / (TP + FN)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)

    # Save metrics and confusion matrix for each model
    model_results_15_features[model_name] = {
        'Accuracy': accuracy,
        'Kappa': kappa,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'ConfusionMatrix': cm
    }

# Convert results to a DataFrame for display
results_df_15_features = pd.DataFrame(model_results_15_features).T

# Print title
print("Results for the 15 features model\n")


# Print results including confusion matrices in a tabular format
headers = ["Model", "Accuracy", "Kappa", "Sensitivity", "Specificity", "Confusion Matrix"]
rows = []

for model_name, result in model_results_15_features.items():
    row = [
        model_name,
        f"{result['Accuracy']:.4f}",
        f"{result['Kappa']:.4f}",
        f"{result['Sensitivity']:.4f}",
        f"{result['Specificity']:.4f}",
        f"\n{result['ConfusionMatrix']}\n"
    ]
    rows.append(row)

# Print the tabulated results
print(tabulate(rows, headers=headers, tablefmt='grid'))
