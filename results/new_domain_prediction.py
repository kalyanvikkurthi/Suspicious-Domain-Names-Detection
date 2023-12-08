import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

from zxcvbn import zxcvbn
import math
from collections import Counter
import statistics
from socket import *
import datetime
import re
import warnings


import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from tabulate import tabulate  # Import the tabulate library

# Suppress FutureWarnings related to is_sparse deprecation
warnings.simplefilter(action='ignore', category=FutureWarning)



####### No of characters #######
def fnd_nch(strn):
    return len(strn)

######## Unique Characters in a String ####
def Uchar(s):
    return len(set(s))

###### Entropy Calculation ######
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())


####### No of Consonants #######
def ConCount(str):
    vowels = 0
    digits = 0
    consonants = 0
    spaces = 0
    symbols = 0
    str = str.lower()
    for i in range(0, len(str)):
        if(str[i] == 'a' or str[i] == 'e' or str[i] == 'i' or str[i] == 'o' or str[i] == 'u'):
            vowels = vowels + 1
        elif((str[i] >= 'a' and str[i] <= 'z')):
            consonants = consonants + 1
        elif( str[i] >= '0' and str[i] <= '9'):
            digits = digits + 1
        elif (str[i] ==' '):
            spaces = spaces + 1
        else:
            symbols = symbols + 1
    return consonants


####### No of Vowels #######

def VowCount(s):
    vowels = 0
    consonants = 0
    for i in s:
        if(i == 'a' or i == 'e' or i == 'i' or i == 'o' or i == 'u' or i == 'A' or i == 'E' or i == 'I' or i == 'O' or i == 'U'):
            vowels = vowels + 1
        else:
            consonants = consonants + 1
    return vowels


####### No of special characters #######
def SpCount(s):
    schar = 0
    for i in s:
        if(i == '_' or i == '-' or i == '.'):
            schar = schar + 1
    return schar


####### No of Numericals characters #######
def NCount(s):
    Num = 0
    for i in s:
        if(i == '0' or i == '1' or i == '2' or i == '3' or i == '4' or i == '5' or i == '6' or i == '7' or i == '8' or i=='9'):
            Num = Num + 1
    return Num
            
##### Count of N-gram Occurences ####
def CountNgram(b,n):
    lst=[b[i:i+n] for i in range(len(b)-n+1)]
    res=dict((x,lst.count(x)) for x in set(lst))
    return res

###### Getting Mean for 1-Gram Occurences ##### 
def getmean1(s):
    ng1_result=CountNgram(s,1)
    mean1 = sum(ng1_result.values()) / len(ng1_result)  
    return mean1  

###### Getting Variance for 1-Gram Occurences ##### 

def getvar1(s):
    ng1_result=CountNgram(s,1)
    ng1_result_list = list(ng1_result.values()) ## Storing values of dictionary in python list format
    return statistics.variance(ng1_result_list)



###### Getting Standard Deviation for 1-Gram Occurences ##### 
def getstdev1(s):
    ng1_result=CountNgram(s,1)
    ng1_result_list = list(ng1_result.values()) ## Storing values of dictionary in python list format
    return statistics.stdev(ng1_result_list)



###### Getting Mean for 2-Gram Occurences ##### 
def getmean2(s):
    ng2_result=CountNgram(s,2)
    ng2_result_list=list(ng2_result.values())
    return statistics.mean(ng2_result_list)  

###### Getting Variance for 2-Gram Occurences ##### 
def getvar2(s):
    ng2_result=CountNgram(s,2)
    ng2_result_list=list(ng2_result.values())
    return statistics.variance(ng2_result_list)  




###### Getting Standard Deviation for 2-Gram Occurences ##### 
def getstdev2(s):
    ng2_result=CountNgram(s,2)
    ng2_result_list=list(ng2_result.values())
    return statistics.stdev(ng2_result_list)  


def listToString(s):  
    str1 = " "  
    return (str1.join(s))

def removesp(string): 
    return "".join(string.split())

def masknames(s):
    strings=s
    cs=listToString([re.sub("b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|x|z|y|w", "c", x) for x in strings])
    str1=removesp(cs)
    #print(str1)
    cs=listToString([re.sub("a|e|i|o|u", "v", x) for x in str1])
    str2=removesp(cs)
    #print(str2)
    cs=listToString([re.sub("0|1|2|3|4|5|6|7|8|9", "n", x) for x in str2])
    str3=removesp(cs)
    #print(str3)
    cs=listToString([re.sub("\\.|\\-|\\_", "s", x) for x in str3])
    str4=removesp(cs)
    return str4

############# cc #########
def get_cc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['cc'])
    except: 
        return(0)

#get_cc("bcd")


############# cv #########
def get_cv(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['cv'])
    except: 
        return(0)

#get_cv("bacd")


############# vc #########
def get_vc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['vc'])
    except: 
        return(0)

#get_vc("abacd")


############# vv #########
def get_vv(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['vv'])
    except: 
        return(0)

#get_vv("abacd")



############# nc #########
def get_nc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['nc'])
    except: 
        return(0)

#get_nc("a9bacd")



############# cn #########
def get_cn(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['cn'])
    except: 
        return(0)

#get_cn("a9b8acd")

############# sc #########
def get_sc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['sc'])
    except: 
        return(0)

#get_sc("a9b8a.cd")



############# cs #########
def get_cs(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['cs'])
    except: 
        return(0)

#get_sc("a9b8ab.cd")


############# nv #########
def get_nv(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['nv'])
    except: 
        return(0)

#get_nv("a9b8ab.cd")


############# vn #########
def get_vn(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,2)
    try:
        return(nres['vn'])
    except: 
        return(0)

#get_vn("a9b8ab.cd")


############# ccc #########
def get_ccc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['ccc'])
    except: 
        return(0)

#get_ccc("cdr")


############# ccc #########
def get_ccc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['ccc'])
    except: 
        return(0)

#get_ccc("cdr")


############# cvc #########
def get_cvc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['cvc'])
    except: 
        return(0)

#get_cvc("car")



############# vcc #########
def get_vcc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['vcc'])
    except: 
        return(0)

#get_vcc("adr")



############# vcv #########
def get_vcv(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['vcv'])
    except: 
        return(0)

#get_vcv("aca")


############# ccv #########
def get_ccv(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['ccv'])
    except: 
        return(0)

#get_ccv("cda")


############# vvv #########
def get_vvv(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['vvv'])
    except: 
        return(0)

#get_vvv("aaa")


############# cvv #########
def get_cvv(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['cvv'])
    except: 
        return(0)
############# vvc #########
def get_vvc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['vvc'])
    except: 
        return(0)

#get_vvc("aab")


############# ncc #########
def get_ncc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['ncc'])
    except: 
        return(0)

#get_ncc("9fb")


############# nvc #########
def get_nvc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['nvc'])
    except: 
        return(0)

#get_nvc("9ab")


############# csc #########
def get_csc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['csc'])
    except: 
        return(0)

#get_csc("b.v")


############# cnc #########
def get_cnc(s):
    Mstring=masknames(s)
    nres=CountNgram(Mstring,3)
    try:
        return(nres['cnc'])
    except: 
        return(0)



def calculate_features_and_update(domain):

    result = zxcvbn(domain)
    guesses = result["guesses"]
    matchedword = result["sequence"][0]["matched_word"] if "matched_word" in result["sequence"][0] else ""
    calc_time = result["calc_time"]
    score = result["score"]
    warning = result["feedback"]["warning"]

    # Calculate your features here
    nch = fnd_nch(domain)
    entropy_val = entropy(domain)
    uchar_val = Uchar(domain)
    concount_val = ConCount(domain)
    vowcount_val = VowCount(domain)
    spcount_val = SpCount(domain)
    ncount_val = NCount(domain)
    mean1_val = getmean1(domain)
    var1_val = getvar1(domain)
    sd1_val = getstdev1(domain)
    mean2_val = getmean2(domain)
    var2_val = getvar2(domain)
    sd2_val = getstdev2(domain)
    cc_val = get_cc(domain)
    cv_val = get_cv(domain)
    vc_val = get_vc(domain)
    vv_val = get_vv(domain)
    nc_val = get_nc(domain)
    cn_val = get_cn(domain)
    sc_val = get_sc(domain)
    cs_val = get_cs(domain)
    nv_val = get_nv(domain)
    vn_val = get_vn(domain)
    ccc_val = get_ccc(domain)
    cvc_val = get_cvc(domain)
    vcc_val = get_vcc(domain)
    vcv_val = get_vcv(domain)
    ccv_val = get_ccv(domain)
    vvv_val = get_vvv(domain)
    cvv_val = get_cvv(domain)
    vvc_val = get_vvc(domain)
    ncc_val = get_ncc(domain)
    nvc_val = get_nvc(domain)
    csc_val = get_csc(domain)
    cnc_val = get_cnc(domain)

    # Create a new DataFrame with the calculated features and other columns
    new_data = pd.DataFrame([[
        domain, guesses, matchedword, calc_time, score, warning,
        nch, entropy_val, uchar_val, concount_val,
        vowcount_val, spcount_val, ncount_val, mean1_val,
        var1_val, sd1_val, mean2_val, var2_val,
        sd2_val, cc_val, cv_val, vc_val,
        vv_val, nc_val, cn_val, sc_val,
        cs_val, nv_val, vn_val, ccc_val,
        cvc_val, vcc_val, vcv_val, ccv_val,
        vvv_val, cvv_val, vvc_val, ncc_val,
        nvc_val, csc_val, cnc_val
    ]], columns=[
        "Domain", "guesses", "matched_word", "calc_time", "score",
        "feedback_warning", "nch", "entropy", "Uchar", "ConCount", "VowCount",
        "SpCount", "NCount", "mean1", "var1", "sd1", "mean2", "var2", "sd2",
        "cc", "cv", "vc", "vv", "nc", "cn", "sc", "cs", "nv", "vn", "ccc",
        "cvc", "vcc", "vcv", "ccv", "vvv", "cvv", "vvc", "ncc", "nvc", "csc", "cnc"
    ])

    # Update 'matched_word' and 'feedback_warning' columns in the new DataFrame
    new_data['matched_word'] = new_data['matched_word'].apply(lambda x: '1' if pd.notna(x) and x != '' else '0').astype('category')
    new_data['feedback_warning'] = new_data['feedback_warning'].apply(lambda x: '1' if pd.notna(x) and x != '' else '0').astype('category')

    X = new_data.drop(columns=['Domain', 'calc_time'])

    return X



# Modify the predict_domain function to include feature selection
def predict_domain(domain, selected_features):
    # Load the trained models for 15 features using joblib each time
    trained_models_15_features = joblib.load('all_models_15_featuresJ.joblib')

    # Assuming calculate_features_and_update returns a DataFrame
    features_df = calculate_features_and_update(domain)

    # Select only the specified features used during model training
    features = features_df[selected_features]

    predictions = {}
    for model_name, model in trained_models_15_features.items():
        predictions[model_name] = model.predict(features)[0]  # Assuming binary classification
    return predictions, features


# Function to predict multiple domains
def predict_domains(domains, selected_features):
    results = []
    for domain in domains:
        predictions, features = predict_domain(domain, selected_features)
        
        # Format features as a table with lines separating rows and columns
        features_table = tabulate(features, headers='keys', tablefmt='grid')

        # Format predictions as a table
        predictions_table = tabulate([[key, value] for key, value in predictions.items()],
                                     headers=["Model", "Prediction"], tablefmt="grid")

        results.append({
            "Domain": domain,
            "Features": features_table,
            "Predictions": predictions_table
        })

    return results


# Example usage with selected_features matching model training
selected_features = ['mean1', 'var1', 'sd1', 'sd2', 'ccc', 'cvc', 'vcc', 'vcv', 'cc', 'vv', 'nch', 'Uchar', 'cc', 'cv', 'vc']

domains_to_predict = [input("Enter your domain name to be tested: ")]

# Example domains to predict
# domains_to_predict = ["github.com", "vgaciwaxmxafsxjt.eu"]

# Get predictions and features for multiple domains
results = predict_domains(domains_to_predict, selected_features)

# Print the results
for result in results:
    print("\nList of 15 Features evaluated for the domain:", result["Domain"])
    print(result["Features"])
    print("\nPredictions for Domain:", result["Domain"])
    print(result["Predictions"])




