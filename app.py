
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
#from tensorflow.keras.models import load_model

st.title("Lung Cancer Prediction")
st.markdown("### Enter Patient Information")

age = st.number_input("Age at Histological Diagnosis", min_value=0, max_value=120)
weight = st.number_input("Weight (lbs)", min_value=0.0, max_value=500.0)
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", ["Asian", "Caucasian", "Latino", "Native Hawaiian/ Pacific Islander"])
pack_years = st.number_input("Pack Years")
days_between_ct_surgery = st.number_input("Days between CT and surgery")
smoking = st.selectbox("Smoking", ["Former", "Current", "Non smoker"])
gg = st.selectbox("%GG", [">0-25%", "25-50%", "50-75%", "75-<100", "100%"])
histology = st.selectbox("Histology", ["Adenocarcinoma", "Squamous cell carcinoma", "NSCLC NOS (not otherwise specified)"])
patho_t_stage = st.selectbox("Pathological T stage", ["T1a", "T3", "T1b", "Tis", "T4", "T2b"])
patho_n_stage = st.selectbox("Pathological N stage", ["N0", "N2", "N1"])
patho_m_stage = st.selectbox("Pathological M stage", ["M0", "M1b", "M1a"])
histopathological_grade = st.selectbox("Histopathological Grade", ["G1 Well differentiated", "G2 Moderately differentiated", "G3 Poorly differentiated", "Other, Type I: Well to moderately differentiated", "Other, Type II: Moderately to poorly differentiated"])
Lymphovascular_invasion = st.selectbox("Lymphovascular Invasion", ["Present", "Absent"])
pleural_invasion = st.selectbox("Pleural invasion (elastic, visceral, or parietal)", ["Yes", "No"])
egfr_mutation = st.selectbox("EGFR mutation status", ["Mutant", "Wildtype", "Unknown"])
kras_mutation = st.selectbox("KRAS mutation status", ["Mutant", "Wildtype", "Unknown"])
alk_translocation = st.selectbox("ALK translocation status", ["Wildtype", "Translocated", "Unknown"])
adjuvant_treatment = st.selectbox("Adjuvant Treatment", ["Yes", "No"])
chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
radiation = st.selectbox("Radiation", ["Yes", "No"])
recurrence = st.selectbox("Recurrence", ["Yes", "No"])
recurrence_location = st.selectbox("Recurrence Location", ["Local", "Regional"])
years_since_quit = st.number_input("Number of years since quitting")

input_data = {
    'Age at Histological Diagnosis': age,
    'Weight (lbs)': weight,
    'Gender': gender,
    'Ethnicity': ethnicity,
    'Pack Years': pack_years,
    'Days between CT and surgery': days_between_ct_surgery,
    'Smoking status': smoking,
    '%GG': gg,
    'Histology': histology,
    'Pathological T stage': patho_t_stage,
    'Pathological N stage': patho_n_stage,
    'Pathological M stage': patho_m_stage,
    'Histopathological Grade': histopathological_grade,
    'Lymphovascular invasion': Lymphovascular_invasion,
    'Pleural invasion (elastic, visceral, or parietal)': pleural_invasion,
    'EGFR mutation status': egfr_mutation,
    'KRAS mutation status': kras_mutation,
    'ALK translocation status': alk_translocation,
    'Adjuvant Treatment': adjuvant_treatment,
    'Chemotherapy': chemotherapy,
    'Radiation': radiation,
    'Recurrence': recurrence,
    'Recurrence Location': recurrence_location,
    'Number of years since quitting': years_since_quit
}

input_df = pd.DataFrame([input_data])
input_df

import pandas as pd
import random

def handle_missing_values(data, column_name, missing_value, strategy="mode"):
    missing_indices = data[column_name] == missing_value

    if strategy == "mode":
        mode_value = data[column_name].mode()[0]
        data.loc[missing_indices, column_name] = mode_value

    elif strategy == "random":
        non_missing_values = data.loc[~missing_indices, column_name].dropna().unique()
        data.loc[missing_indices, column_name] = [
            random.choice(non_missing_values) for _ in range(missing_indices.sum())
        ]

    elif strategy == "new_category":
        data.loc[missing_indices, column_name] = "Unknown"

    else:
        raise ValueError("Invalid strategy! Choose from 'mode', 'random', or 'new_category'.")

    return data


#handle_missing_values(data, 'Ethnicity', 'Not Recorded In Database', strategy="mode")
handle_missing_values(input_df, '%GG', 'Not Assessed', strategy="mode")
handle_missing_values(input_df, 'Pathological T stage', 'Not Collected', strategy="random")
handle_missing_values(input_df, 'Pathological N stage', 'Not Collected', strategy="random")
handle_missing_values(input_df, 'Pathological M stage', 'Not Collected', strategy="random")
handle_missing_values(input_df, 'Histopathological Grade', 'Not Collected', strategy="random")
handle_missing_values(input_df, 'Lymphovascular invasion', 'Not Collected', strategy="random")
handle_missing_values(input_df, 'Pleural invasion (elastic, visceral, or parietal)', 'Not Collected', strategy="random")
handle_missing_values(input_df, 'EGFR mutation status', 'Not Collected', strategy="mode")
handle_missing_values(input_df, 'KRAS mutation status', 'Not Collected', strategy="mode")
handle_missing_values(input_df, 'ALK translocation status', 'Not Collected', strategy="mode")
handle_missing_values(input_df, 'Chemotherapy', 'Not Collected', strategy="mode")
handle_missing_values(input_df, 'Radiation', 'Not Collected', strategy="mode")
handle_missing_values(input_df, 'Recurrence', 'Not Collected', strategy="mode")
handle_missing_values(input_df, 'KRAS mutation status', 'Not Collected', strategy="mode")
handle_missing_values(input_df, 'KRAS mutation status', 'Not Collected', strategy="mode")
handle_missing_values(input_df, 'Adjuvant Treatment', 'Not Collected', strategy="mode")


def onehotEncoder(data, category_feature):
    one_hot_encoder = OneHotEncoder(drop='first')
    encoded_features = one_hot_encoder.fit_transform(data[[category_feature]])

    if encoded_features.shape[1] == 0:
        print(f"No features generated for {category_feature}, check the input data.")
        return data  # Skip if no new columns were created

    encoded_df = pd.DataFrame(encoded_features.toarray(),
                              columns=one_hot_encoder.get_feature_names_out([category_feature]),
                              index=data.index)

    data = data.drop(category_feature, axis=1).join(encoded_df)
    return data


input_df.columns = input_df.columns.str.strip()
columns_to_encode = ['Gender', 'Smoking status', 'Ethnicity', '%GG', 'Histology',
                     'Pathological T stage', 'Pathological N stage',
                     'Pathological M stage', 'Histopathological Grade', 'Adjuvant Treatment',
                     'Lymphovascular invasion','Pleural invasion (elastic, visceral, or parietal)',
                     'EGFR mutation status', 'KRAS mutation status', 'ALK translocation status',
                     'Chemotherapy', 'Radiation', 'Recurrence', 'Recurrence Location']

for feature in columns_to_encode:
    if feature in input_df.columns:
        input_df = onehotEncoder(input_df, feature)
    else:
        print(f"Column '{feature}' not found in DataFrame.")



import pandas as pd

def impute_missing_numerical(data, column_name, missing_value, strategy="mean", fill_value=None):
    data[column_name] = pd.to_numeric(data[column_name].replace(missing_value, pd.NA), errors='coerce')

    if strategy == "mean":
        fill_value = data[column_name].mean()
    elif strategy == "median":
        fill_value = data[column_name].median()
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("Please provide a fill_value when using the 'constant' strategy.")
    else:
        raise ValueError("Invalid strategy! Choose from 'mean', 'median', or 'constant'.")
    data[column_name].fillna(fill_value, inplace=True)

    return data


impute_missing_numerical(input_df, 'Weight (lbs)', 'Not Collected', strategy="mean", fill_value=None)
impute_missing_numerical(input_df, 'Pack Years', 'Not Collected', strategy="mean", fill_value=None)

from datetime import datetime

def years_since_stopped_smoking(data, column_name):
    current_year = datetime.now().year

    data[f'years_since_Quit Smoking Year'] = current_year - pd.to_numeric(data[column_name], errors='coerce')

    return data

years_since_stopped_smoking(input_df, 'Number of years since quitting')

data = input_df.drop('Number of years since quitting', axis=1, errors='ignore')

# List of expected columns (replace this with the exact list your model expects)
expected_columns = [
    'Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Days between CT and surgery',
    'Gender_Male', 'Smoking status_Former', 'Smoking status_Nonsmoker', 'Ethnicity_Asian',
    'Ethnicity_Caucasian', 'Ethnicity_Hispanic/Latino', 'Ethnicity_Native Hawaiian/Pacific Islander',
    '%GG_100%', '%GG_25 - 50%', '%GG_50 - 75%', '%GG_75 - < 100%', '%GG_>0 - 25%',
    'Histology_NSCLC NOS (not otherwise specified)', 'Histology_Squamous cell carcinoma',
    'Pathological T stage_T1b', 'Pathological T stage_T2a', 'Pathological T stage_T2b',
    'Pathological T stage_T3', 'Pathological T stage_T4', 'Pathological T stage_Tis',
    'Pathological N stage_N1', 'Pathological N stage_N2', 'Pathological M stage_M1a',
    'Pathological M stage_M1b', 'Histopathological Grade_G2 Moderately differentiated',
    'Histopathological Grade_G3 Poorly differentiated', 'Histopathological Grade_Other, Type I: Well to moderately differentiated',
    'Histopathological Grade_Other, Type II: Moderately to poorly differentiated', 'Adjuvant Treatment_Yes',
    'Lymphovascular invasion_Present', 'Pleural invasion (elastic, visceral, or parietal)_Not collected',
    'Pleural invasion (elastic, visceral, or parietal)_Yes', 'EGFR mutation status_Not collected',
    'EGFR mutation status_Unknown', 'EGFR mutation status_Wildtype', 'KRAS mutation status_Not collected',
    'KRAS mutation status_Unknown', 'KRAS mutation status_Wildtype', 'ALK translocation status_Translocated',
    'ALK translocation status_Unknown', 'ALK translocation status_Wildtype', 'Chemotherapy_Yes',
    'Radiation_Yes', 'Recurrence_no', 'Recurrence_yes', 'Recurrence Location_local',
    'Recurrence Location_regional', 'Recurrence Location_nan', 'Survival Status_Dead', 'years_since_Quit Smoking Year'
]

# Ensure all expected columns are present
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match model input
input_df = input_df[expected_columns]


new_column_names = {
    '%GG_100%': '%GG_100%',
    '%GG_>0 - 25%': '%GG_0 - 25%',
    '%GG_25 - 50%': '%GG_25 - 50%',
    '%GG_50 - 75%':  '%GG_50 - 75%',
    '%GG_75 - < 100%': '%GG_75 - 100%'

}

data = input_df.rename(columns=new_column_names)

input_df = input_df.apply(pd.to_numeric, errors='coerce')
input_df = input_df.fillna(0)
input_df = input_df.astype(float)




if st.button("Predict"):
    model = tf.keras.models.load_model("my_model.h5")
    pred = (model.predict(input_df) > 0.5).astype("int32")
    if pred == 1:
        st.success("High likelihood of lung cancer.")
    else:
        st.success("Low likelihood of lung cancer.")
