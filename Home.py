# Developed by Hikmet Can Çubukçu

import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

st.set_page_config(
    page_title="Outcome prediction after allo-HSCT",
    page_icon="🚥",
    layout="wide"
)

with st.sidebar:
    choice = st.radio("**:blue[Choose the prediction model]**" , ["Acute GVHD prediction", "Moderate to very severe acute GVHD prediction",
                      "Survival prediction", "Chronic GVHD prediction"])
    #st.image('./images/QC Constellation icon.png')
    #st.markdown('---')
    st.info('*Developed by Hikmet Can Çubukçu, MD, MSc, EuSpLM* <hikmetcancubukcu@gmail.com>')
    
instructions = """
**Enter the variables below to predict the outcome of your interest**
"""


@st.cache_resource
def get_model(model):
    return load_model(f"./ML_models/{model}")

def predict_target(model, data):
    prediction = predict_model(model, data = data)
    return prediction['Label'] # to be accessed the target variable

def predict_target(model, data, pred_name):
    # Use PyCaret to make the prediction
    prediction = predict_model(model, data=data)
    
    # Try to extract the predicted value from the 'Label' column or any other column that holds predictions
    if 'Label' in prediction.columns:
        predicted_value = prediction['Label'].values[0]
    else:
        # If 'Label' doesn't exist, try another likely column
        predicted_value_1 = prediction[prediction.columns[-2]].values[0]  # Last column typically holds predictions
        if predicted_value_1 == 0:
            predicted_value_1 = f'{pred_name} is not expected'
        else:
            predicted_value_1 = f'{pred_name} is expected'
        
        predicted_value_2 = prediction[prediction.columns[-1]].values[0]  # Last column typically holds predictions
        predicted_value_2 = predicted_value_2*100
    # Return both the input features and the predicted target value
    if predicted_value_1 == f'{pred_name} is not expected':
        return st.info(f'**Pediction result:** {predicted_value_1} with a probability of {round(predicted_value_2,2)}%')
    else:
        return st.error(f'**Pediction result:** {predicted_value_1} with a probability of {round(predicted_value_2,2)}%')
        
    #return predicted_value_1, predicted_value_2   # pd.DataFrame(prediction)

if choice == "Acute GVHD prediction":
    pred_name = "Acute GVHD"
    st.header(":blue[Acute GVHD prediction model]")
    model_name = "aGVHD_catboost_model"
    model = get_model(model_name)
    st.write(" ")
    st.markdown(instructions)
    
    input_variables = model.feature_names_
    # Input vars Create a DataFrame for better visualization
    df_input_variables = pd.DataFrame(input_variables, columns=["Input Variables"])
        
    # Load the uploaded data
    uploaded_data = pd.read_excel('tez_selected_data_v2_clean_v6_cat.xlsx')

    # Collect raw input variables (before feature engineering)

    # HLA_numeric_conformance
    HLA_numeric_conformance = None # st.number_input("HLA numeric conformance", min_value=0.0, value=0.0)

    # Haploidentical transplant
    Haploidentical_transplant = None # st.checkbox("Haploidentical transplant")

    # Recipent age
    Recipent_age = None # st.number_input("Recipent age", min_value=0, max_value=120, value=30)

    # Donor age
    Donor_age = st.number_input("Donor age", min_value=0, max_value=120, value=30)

    # Recipent gender
    Recipent_gender = st.selectbox("Recipent gender", options=uploaded_data['Recipent gender'].dropna().unique())

    # Donor gender
    Donor_gender = st.selectbox("Donor gender", options=uploaded_data['Donor gender'].dropna().unique())

    # Gender mismatch
    if Recipent_gender == Donor_gender:
        Gender_mismatch = False
    else:
        Gender_mismatch = True

    # Main diagnosis
    Main_diagnosis = st.selectbox("Main diagnosis", options=uploaded_data['Main diagnosis'].dropna().unique())

    # Diagnosis
    Diagnosis = st.selectbox("Diagnosis", options=uploaded_data['Diagnosis'].dropna().unique())

    # Stem cell source
    Stem_cell_source = st.selectbox("Stem cell source", options=uploaded_data['Stem cell source'].dropna().unique())

    # Conditioning therapy
    Conditioning_therapy = st.selectbox("Conditioning therapy", options=uploaded_data['Conditioning therapy'].dropna().unique())

    # Conditioning therapy type
    Conditioning_therapy_type = st.selectbox("Conditioning therapy type", options=uploaded_data['Conditioning therapy type'].dropna().unique())

    # Graft-Versus-Host Disease (GVHD) Prophylaxis
    GVHD_Prophylaxis = st.selectbox("Graft-Versus-Host Disease (GVHD) Prophylaxis", options=uploaded_data['Graft-Versus-Host Disease (GVHD) Prophylaxis'].dropna().unique())

    # Recipient blood type
    Recipient_blood_type = st.selectbox("Recipient blood type", options=uploaded_data['Recipient blood type '].dropna().unique())

    # Donor blood type
    Donor_blood_type = st.selectbox("Donor blood type", options=uploaded_data['Recipient blood type '].dropna().unique())
    
    # Blood type mismatch
    if Recipient_blood_type == Donor_blood_type:
        Blood_type_mismatch = False
    else:
        Blood_type_mismatch = True


    # Mononuclear cells 10E8
    Mononuclear_cells_10E8 = None # st.number_input("Mononuclear cells 10E8", min_value=0.0, value=0.0)

    # CD34-positive cells 10E6
    CD34_positive_cells_10E6 = None # st.number_input("CD34-positive cells 10E6", min_value=0.0, value=0.0)

    # CD3-positive cells 10E7
    CD3_positive_cells_10E7 = None # st.number_input("CD3-positive cells 10E7", min_value=0.0, value=0.0)

    # CD3-16-56+ 10E7
    CD3_16_56_positive_cells_10E7 = None # st.number_input("CD3-16-56+ cells 10E7", min_value=0.0, value=0.0)

    # CD3+4+ 10E7
    CD3_4_positive_cells_10E7 = None # st.number_input("CD3+4+ cells 10E7", min_value=0.0, value=0.0)

    # CD3+8+ 10E7
    CD3_8_positive_cells_10E7 = None # st.number_input("CD3+8+ cells 10E7", min_value=0.0, value=0.0)

    # CD19+ 10E7
    CD19_positive_cells_10E7 = None # st.number_input("CD19+ cells 10E7", min_value=0.0, value=0.0)

    # HSV infection status
    HSV_infection_status = st.checkbox("HSV infection status")

    # Engraftment status
    Engraftment_status = st.checkbox("Engraftment status")

    # Number of donor lymphocyte infusion
    Number_of_donor_lymphocyte_infusion = st.number_input("Number of donor lymphocyte infusion", min_value=0, value=0)

    # Additional variables based on your request

    # Mismatched locus count
    Mismatched_locus_count = st.number_input("Mismatched locus count", min_value=0, max_value=2, value=0)



    # Total Body Irradiation Status
    Total_Body_Irradiation_Status = st.checkbox("Total Body Irradiation")

    # Number of afferes applied to donor
    Number_of_afferes_applied_to_donor = st.number_input("Number of afferes applied to donor", min_value=0, max_value=5, value=1)

    # The amount of graft given to the recipient
    The_amount_of_graft_given_to_the_recipient = st.number_input("The amount of graft given to the recipient", min_value=0.0, max_value=uploaded_data['The amount of graft given to the recipient'].max(), value=0.0)

    # Total Nucleated Cells 10E8
    Total_Nucleated_Cells_10E8 = st.number_input("Total Nucleated Cells 10E8", min_value=0.0, max_value=uploaded_data['Total Nucleated Cells 10E8'].max(), value=0.0)

    # CMV infection status
    CMV_infection_status = st.checkbox("CMV infection")

    # Presence of hemorrhagic cystitis
    Presence_of_hemorrhagic_cystitis = st.checkbox("Presence of hemorrhagic cystitis")
    if Presence_of_hemorrhagic_cystitis:
        Hemorrhagic_cystitis_grade = st.number_input("Hemorrhagic cystitis grade", min_value=1, max_value=4, value=1)
    else:
        Hemorrhagic_cystitis_grade = 0

    # The day the neutrophil count exceeds 1000
    The_day_the_neutrophil_count_exceeds_1000 = st.number_input("The day the neutrophil count exceeds 1000", min_value=0, max_value=145, value=0)

    # Number of relapse
    Number_of_relapse = st.number_input("Number of relapse", min_value=0, max_value=3, value=0)

    # Collect inputs into a dictionary
    input_data = {
        "HLA_numeric_conformance": HLA_numeric_conformance,
        "Haploidentical transplant": Haploidentical_transplant,
        "Recipent age": Recipent_age,
        "Main diagnosis": Main_diagnosis,
        "Diagnosis": Diagnosis,
        "Stem cell source": Stem_cell_source,
        "Conditioning therapy": Conditioning_therapy,
        "Conditioning therapy type": Conditioning_therapy_type,
        "Graft-Versus-Host Disease (GVHD) Prophylaxis": GVHD_Prophylaxis,
        "Recipient blood type ": Recipient_blood_type,
        "Donor blood type": Donor_blood_type,
        "Gender mismatch": Gender_mismatch,
        "Mononuclear cells 10E8": Mononuclear_cells_10E8,
        "CD34-positive cells 10E6": CD34_positive_cells_10E6,
        "CD3-positive cells 10E7": CD3_positive_cells_10E7,
        "CD3-16-56+ 10E7": CD3_16_56_positive_cells_10E7,
        "CD3+4+ 10E7": CD3_4_positive_cells_10E7,
        "CD3+8+ 10E7": CD3_8_positive_cells_10E7,
        "CD19+ 10E7": CD19_positive_cells_10E7,
        "HSV infection status": HSV_infection_status,
        "Engraftment status": Engraftment_status,
        "Number of donor lymphocyte infusion": Number_of_donor_lymphocyte_infusion,
        "Mismatched_locus_count": Mismatched_locus_count,
        "Donor age": Donor_age,
        "Total Body Irradiation Status": Total_Body_Irradiation_Status,
        "Blood type mismatch": Blood_type_mismatch,
        "Recipent gender": Recipent_gender,
        "Donor gender": Donor_gender,
        "Number of afferes applied to donor": Number_of_afferes_applied_to_donor,
        "The amount of graft given to the recipient": The_amount_of_graft_given_to_the_recipient,
        "Total Nucleated Cells 10E8": Total_Nucleated_Cells_10E8,
        "CMV infection status": CMV_infection_status,
        "Presence of hemorrhagic cystitis": Presence_of_hemorrhagic_cystitis,
        "Hemorrhagic cystitis grade": Hemorrhagic_cystitis_grade,
        "The day the neutrophil count exceeds 1000": The_day_the_neutrophil_count_exceeds_1000,
        "Number of relapse": Number_of_relapse
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    #input_df
    
    predict_button_v1 = st.button("Predict")
    
    if predict_button_v1:
        output = predict_target(model, input_df, pred_name)
        
        #st.info(f'Prediction: {output}')
        # st.dataframe(output)

    #pass

if choice == "Moderate to very severe acute GVHD prediction":
    pred_name = "Moderate to very severe acute GVHD"
    st.header(":blue[Moderate to very severe acute GVHD prediction model]")
    model_name = "aGVHD_severity_tuned_rf_model"
    model = get_model(model_name)
    st.write(" ")
    st.markdown(instructions)
        
    # Load the uploaded data
    uploaded_data = pd.read_excel('tez_selected_data_v2_clean_v6_cat.xlsx')

    # Collect raw input variables (before feature engineering)

    # HLA_numeric_conformance
    HLA_numeric_conformance = None # st.number_input("HLA numeric conformance", min_value=0.0, value=0.0)

    # Haploidentical transplant
    Haploidentical_transplant = None # st.checkbox("Haploidentical transplant")

    # Recipent age
    Recipent_age = None # st.number_input("Recipent age", min_value=0, max_value=120, value=30)

    # Donor age
    Donor_age = st.number_input("Donor age", min_value=0, max_value=120, value=30)  

    # Recipent gender
    Recipent_gender = st.selectbox("Recipent gender", options=uploaded_data['Recipent gender'].dropna().unique())

    # Donor gender
    Donor_gender = st.selectbox("Donor gender", options=uploaded_data['Donor gender'].dropna().unique())

    # Gender mismatch
    if Recipent_gender == Donor_gender:
        Gender_mismatch = False
    else:
        Gender_mismatch = True

    # Gender_mismatch = st.checkbox("Gender mismatch")
    
    # Main diagnosis
    Main_diagnosis = st.selectbox("Main diagnosis", options=uploaded_data['Main diagnosis'].dropna().unique())

    # Diagnosis
    Diagnosis = st.selectbox("Diagnosis", options=uploaded_data['Diagnosis'].dropna().unique())

    # Stem cell source
    Stem_cell_source = st.selectbox("Stem cell source", options=uploaded_data['Stem cell source'].dropna().unique())

    # Conditioning therapy
    Conditioning_therapy = st.selectbox("Conditioning therapy", options=uploaded_data['Conditioning therapy'].dropna().unique())

    # Conditioning therapy type
    Conditioning_therapy_type = st.selectbox("Conditioning therapy type", options=uploaded_data['Conditioning therapy type'].dropna().unique())

    # Graft-Versus-Host Disease (GVHD) Prophylaxis
    GVHD_Prophylaxis = st.selectbox("Graft-Versus-Host Disease (GVHD) Prophylaxis", options=uploaded_data['Graft-Versus-Host Disease (GVHD) Prophylaxis'].dropna().unique())

    # Recipient blood type
    Recipient_blood_type = st.selectbox("Recipient blood type", options=uploaded_data['Recipient blood type '].dropna().unique())

    # Donor blood type
    Donor_blood_type = st.selectbox("Donor blood type", options=uploaded_data['Recipient blood type '].dropna().unique())
    
    # Blood type mismatch
    if Recipient_blood_type == Donor_blood_type:
        Blood_type_mismatch = False
    else:
        Blood_type_mismatch = True
    

    # Mononuclear cells 10E8
    Mononuclear_cells_10E8 = st.number_input("Mononuclear cells 10E8", min_value=0.0, value=0.0)

    # CD34-positive cells 10E6
    CD34_positive_cells_10E6 = st.number_input("CD34-positive cells 10E6", min_value=0.0, value=0.0)

    # CD3-positive cells 10E7
    CD3_positive_cells_10E7 = None # st.number_input("CD3-positive cells 10E7", min_value=0.0, value=0.0)

    # CD3-16-56+ 10E7
    CD3_16_56_positive_cells_10E7 = st.number_input("CD3-16-56+ cells 10E7", min_value=0.0, value=0.0)

    # CD3+4+ 10E7
    CD3_4_positive_cells_10E7 = st.number_input("CD3+4+ cells 10E7", min_value=0.0, value=0.0)

    # CD3+8+ 10E7
    CD3_8_positive_cells_10E7 = st.number_input("CD3+8+ cells 10E7", min_value=0.0, value=0.0)

    # CD19+ 10E7
    CD19_positive_cells_10E7 = st.number_input("CD19+ cells 10E7", min_value=0.0, value=0.0)
    
    # Total Nucleated Cells 10E8
    Total_Nucleated_Cells_10E8 = st.number_input("Total Nucleated Cells 10E8", min_value=0.0, max_value=uploaded_data['Total Nucleated Cells 10E8'].max(), value=0.0)

    # HSV infection status
    HSV_infection_status = None #st.checkbox("HSV infection")

    # Engraftment status
    Engraftment_status = None # st.checkbox("Engraftment status")

    # Number of donor lymphocyte infusion
    Number_of_donor_lymphocyte_infusion = st.number_input("Number of donor lymphocyte infusion", min_value=0, value=0)

    # Additional variables based on your request

    # Mismatched locus count
    Mismatched_locus_count = st.number_input("Mismatched locus count", min_value=0, max_value=2, value=0)

    # Total Body Irradiation Status
    Total_Body_Irradiation_Status = st.checkbox("Total Body Irradiation")

    # Number of afferes applied to donor
    Number_of_afferes_applied_to_donor = None # st.number_input("Number of afferes applied to donor", min_value=0, max_value=5, value=1)

    # The amount of graft given to the recipient
    The_amount_of_graft_given_to_the_recipient = st.number_input("The amount of graft given to the recipient", min_value=0.0, max_value=uploaded_data['The amount of graft given to the recipient'].max(), value=0.0)

    # CMV infection status
    CMV_infection_status = st.checkbox("CMV infection")

    # Presence of hemorrhagic cystitis
    Presence_of_hemorrhagic_cystitis = st.checkbox("Presence of hemorrhagic cystitis")
    if Presence_of_hemorrhagic_cystitis:
        Hemorrhagic_cystitis_grade = st.number_input("Hemorrhagic cystitis grade", min_value=1, max_value=4, value=1)
    else:
        Hemorrhagic_cystitis_grade = 0

    # The day the neutrophil count exceeds 1000
    The_day_the_neutrophil_count_exceeds_1000 = st.number_input("The day the neutrophil count exceeds 1000", min_value=0, max_value=145, value=0)

    # Number of relapse
    Number_of_relapse = st.number_input("Number of relapse", min_value=0, max_value=3, value=0)

    # Collect inputs into a dictionary
    input_data = {
        "HLA_numeric_conformance": HLA_numeric_conformance,
        "Haploidentical transplant": Haploidentical_transplant,
        "Recipent age": Recipent_age,
        "Main diagnosis": Main_diagnosis,
        "Diagnosis": Diagnosis,
        "Stem cell source": Stem_cell_source,
        "Conditioning therapy": Conditioning_therapy,
        "Conditioning therapy type": Conditioning_therapy_type,
        "Graft-Versus-Host Disease (GVHD) Prophylaxis": GVHD_Prophylaxis,
        "Recipient blood type ": Recipient_blood_type,
        "Donor blood type": Donor_blood_type,
        "Gender mismatch": Gender_mismatch,
        "Mononuclear cells 10E8": Mononuclear_cells_10E8,
        "CD34-positive cells 10E6": CD34_positive_cells_10E6,
        "CD3-positive cells 10E7": CD3_positive_cells_10E7,
        "CD3-16-56+ 10E7": CD3_16_56_positive_cells_10E7,
        "CD3+4+ 10E7": CD3_4_positive_cells_10E7,
        "CD3+8+ 10E7": CD3_8_positive_cells_10E7,
        "CD19+ 10E7": CD19_positive_cells_10E7,
        "HSV infection status": HSV_infection_status,
        "Engraftment status": Engraftment_status,
        "Number of donor lymphocyte infusion": Number_of_donor_lymphocyte_infusion,
        "Mismatched_locus_count": Mismatched_locus_count,
        "Donor age": Donor_age,
        "Total Body Irradiation Status": Total_Body_Irradiation_Status,
        "Blood type mismatch": Blood_type_mismatch,
        "Recipent gender": Recipent_gender,
        "Donor gender": Donor_gender,
        "Number of afferes applied to donor": Number_of_afferes_applied_to_donor,
        "The amount of graft given to the recipient": The_amount_of_graft_given_to_the_recipient,
        "Total Nucleated Cells 10E8": Total_Nucleated_Cells_10E8,
        "CMV infection status": CMV_infection_status,
        "Presence of hemorrhagic cystitis": Presence_of_hemorrhagic_cystitis,
        "Hemorrhagic cystitis grade": Hemorrhagic_cystitis_grade,
        "The day the neutrophil count exceeds 1000": The_day_the_neutrophil_count_exceeds_1000,
        "Number of relapse": Number_of_relapse
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    #input_df
    
    predict_button_v1 = st.button("Predict")
    
    if predict_button_v1:
        output = predict_target(model, input_df, pred_name)
        
        #st.info(f'Prediction: {output}')
        # st.dataframe(output)

    #pass    
    

if choice == "Survival prediction":
    st.header(":blue[Survival prediction model]")
    pred_name = "Survival"
    model_name = "surv_catboost_model"
    model = get_model(model_name)
    st.write(" ")
    st.markdown(instructions)
        
    # Load the uploaded data
    uploaded_data = pd.read_excel('tez_selected_data_v2_clean_v6_cat.xlsx')

    # Collect raw input variables (before feature engineering)

    # HLA_numeric_conformance
    HLA_numeric_conformance = st.number_input("HLA numeric conformance", min_value=0.0, value=0.0)
    
    # Mismatched locus count
    Mismatched_locus_count = st.number_input("Mismatched locus count", min_value=0, max_value=2, value=0)
    
    # Haploidentical transplant
    Haploidentical_transplant = st.checkbox("Haploidentical transplant")

    # Recipent age
    Recipent_age = st.number_input("Recipent age", min_value=0, max_value=120, value=30)

    # Donor age
    Donor_age = st.number_input("Donor age", min_value=0, max_value=120, value=30)

    # Main diagnosis
    Main_diagnosis = st.selectbox("Main diagnosis", options=uploaded_data['Main diagnosis'].dropna().unique())

    # Diagnosis
    Diagnosis = st.selectbox("Diagnosis", options=uploaded_data['Diagnosis'].dropna().unique())

    # Stem cell source
    Stem_cell_source = st.selectbox("Stem cell source", options=uploaded_data['Stem cell source'].dropna().unique())

    # Conditioning therapy
    Conditioning_therapy = st.selectbox("Conditioning therapy", options=uploaded_data['Conditioning therapy'].dropna().unique())

    # Conditioning therapy type
    Conditioning_therapy_type = st.selectbox("Conditioning therapy type", options=uploaded_data['Conditioning therapy type'].dropna().unique())

    # Graft-Versus-Host Disease (GVHD) Prophylaxis
    GVHD_Prophylaxis = st.selectbox("Graft-Versus-Host Disease (GVHD) Prophylaxis", options=uploaded_data['Graft-Versus-Host Disease (GVHD) Prophylaxis'].dropna().unique())

    # Recipient blood type
    Recipient_blood_type = st.selectbox("Recipient blood type", options=uploaded_data['Recipient blood type '].dropna().unique())

    # Donor blood type
    Donor_blood_type = st.selectbox("Donor blood type", options=uploaded_data['Recipient blood type '].dropna().unique())
    
    # Blood type mismatch
    if Recipient_blood_type == Donor_blood_type:
        Blood_type_mismatch = False
    else:
        Blood_type_mismatch = True
    
    # HSV infection status
    HSV_infection_status = st.checkbox("HSV infection")

    # Engraftment status
    Engraftment_status = st.checkbox("Engraftment")

    # Total Body Irradiation Status
    Total_Body_Irradiation_Status = st.checkbox("Total Body Irradiation")

    # Number of donor lymphocyte infusion
    Number_of_donor_lymphocyte_infusion = st.number_input("Number of donor lymphocyte infusion", min_value=0, value=0)

    # Additional variables based on your request
    # Recipent gender
    Recipent_gender = st.selectbox("Recipent gender", options=uploaded_data['Recipent gender'].dropna().unique())

    # Donor gender
    Donor_gender = st.selectbox("Donor gender", options=uploaded_data['Donor gender'].dropna().unique())
    
    # Gender mismatch
    if Recipent_gender == Donor_gender:
        Gender_mismatch = False # st.checkbox("Gender mismatch")
    else:
        Gender_mismatch = True
        
    # Number of afferes applied to donor
    Number_of_afferes_applied_to_donor = st.number_input("Number of afferes applied to donor", min_value=0, max_value=5, value=1)

    # The amount of graft given to the recipient
    The_amount_of_graft_given_to_the_recipient = st.number_input("The amount of graft given to the recipient", min_value=0.0, max_value=uploaded_data['The amount of graft given to the recipient'].max(), value=0.0)


    # CMV infection status
    CMV_infection_status = st.checkbox("CMV infection")

    # Presence of hemorrhagic cystitis
    Presence_of_hemorrhagic_cystitis = st.checkbox("Presence of hemorrhagic cystitis")
    if Presence_of_hemorrhagic_cystitis:
        Hemorrhagic_cystitis_grade = st.number_input("Hemorrhagic cystitis grade", min_value=1, max_value=4, value=1)
    else:
        Hemorrhagic_cystitis_grade = 0
    
    Defibrotide_prophylaxis = st.checkbox("Defibrotide prophylaxis")
    
    # The day the neutrophil count exceeds 1000
    The_day_the_neutrophil_count_exceeds_1000 = st.number_input("The day the neutrophil count exceeds 1000", min_value=0, max_value=145, value=0)

    # Additional variables

    The_day_the_platelet_count_exceeds_20000 = st.number_input("The day the platelet count exceeds 20000", min_value=0, max_value=145, value=0)
    The_day_the_platelet_count_exceeds_50000 = st.number_input("The day the platelet count exceeds 50000", min_value=0, max_value=145, value=0)
    
    # Number of relapse
    Number_of_relapse = st.number_input("Number of relapse", min_value=0, max_value=3, value=0)
    
    AGVH_presence = st.checkbox("AGVH presence")

    if AGVH_presence:
        AGVHD_grade = st.number_input("Acute GVHD grade", min_value=1, max_value=4, value=1)
        AGVHD = AGVHD_grade
        if AGVHD_grade == 1:
            AGVH_severity_1vs234 = 0
            AGVH_severity = 0
        elif AGVHD_grade == 2:
            AGVH_severity_1vs234 = 1
            AGVH_severity = 0        
        else:
            AGVH_severity_1vs234 = 1
            AGVH_severity = 1    
        # AGVH_severity_1vs234 = st.checkbox("AGVH severity grade 1 vs. 2,3,4") # st.selectbox("AGVH severity 1 vs 234", options=uploaded_data['AGVH_severity_1vs234'].dropna().unique())
        # AGVH_severity = st.checkbox("AGVH severity grade 1,2 vs. 3,4") # st.selectbox("AGVH severity", options=uploaded_data['AGVH_severity'].dropna().unique())
    else:
        AGVHD = 0
        AGVH_severity_1vs234 = 0
        AGVH_severity = 0
    # AGVHD = st.checkbox("AGVHD")
    # AGVH_severity_1vs234 = st.checkbox("AGVH severity grade 1 vs. 2,3,4") # st.selectbox("AGVH severity 1 vs 234", options=uploaded_data['AGVH_severity_1vs234'].dropna().unique())
    # AGVH_severity = st.checkbox("AGVH severity grade 1,2 vs. 3,4") # st.selectbox("AGVH severity", options=uploaded_data['AGVH_severity'].dropna().unique())
    
    KGVHD_presence = st.checkbox("KGVHD presence")
    
    # Collect inputs into a dictionary
    input_data = {
        "HLA_numeric_conformance": HLA_numeric_conformance,
        "Haploidentical transplant": Haploidentical_transplant,
        "Recipent age": Recipent_age,
        "Main diagnosis": Main_diagnosis,
        "Diagnosis": Diagnosis,
        "Stem cell source": Stem_cell_source,
        "Conditioning therapy": Conditioning_therapy,
        "Conditioning therapy type": Conditioning_therapy_type,
        "Graft-Versus-Host Disease (GVHD) Prophylaxis": GVHD_Prophylaxis,
        "Recipient blood type ": Recipient_blood_type,
        "Donor blood type": Donor_blood_type,
        "Gender mismatch": Gender_mismatch,
        "HSV infection status": HSV_infection_status,
        "Engraftment status": Engraftment_status,
        "Number of donor lymphocyte infusion": Number_of_donor_lymphocyte_infusion,
        "Mismatched_locus_count": Mismatched_locus_count,
        "Donor age": Donor_age,
        "Total Body Irradiation Status": Total_Body_Irradiation_Status,
        "Blood type mismatch": Blood_type_mismatch,
        "Recipent gender": Recipent_gender,
        "Donor gender": Donor_gender,
        "Number of afferes applied to donor": Number_of_afferes_applied_to_donor,
        "The amount of graft given to the recipient": The_amount_of_graft_given_to_the_recipient,
        "CMV infection status": CMV_infection_status,
        "Presence of hemorrhagic cystitis": Presence_of_hemorrhagic_cystitis,
        "Hemorrhagic cystitis grade": Hemorrhagic_cystitis_grade,
        "The day the neutrophil count exceeds 1000": The_day_the_neutrophil_count_exceeds_1000,
        "Number of relapse": Number_of_relapse,
        # additional
        "Defibrotide prophylaxis": Defibrotide_prophylaxis,
        "The day the platelet count exceeds 20000": The_day_the_platelet_count_exceeds_20000,
        "The day the platelet count exceeds 50000": The_day_the_platelet_count_exceeds_50000,
        "AGVHD": AGVHD,
        "AGVH_severity_1vs234": AGVH_severity_1vs234,
        "AGVH_severity": AGVH_severity,
        "AGVH_presence": AGVH_presence,
        "KGVHD_presence": KGVHD_presence
        }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    #input_df
    
    predict_button_v1 = st.button("Predict")
    
    if predict_button_v1:
        output = predict_target(model, input_df, pred_name)
        
    # pass

if choice == "Chronic GVHD prediction":
    st.header(":blue[Chronic GVHD prediction model]")
    pred_name = "Chronic GVHD"
    model_name = "cGVHD_tuned_rf_model"
    model = get_model(model_name)
    st.write(" ")
    st.markdown(instructions)
        
    # Load the uploaded data
    uploaded_data = pd.read_excel('tez_selected_data_v2_clean_v6_cat.xlsx')

    # Collect raw input variables (before feature engineering)

    # HLA_numeric_conformance
    HLA_numeric_conformance = None # st.number_input("HLA numeric conformance", min_value=0.0, value=0.0)

    # Haploidentical transplant
    Haploidentical_transplant = None # st.checkbox("Haploidentical transplant")

    # Recipent age
    Recipent_age = st.number_input("Recipent age", min_value=0, max_value=120, value=30)

    # Donor age
    Donor_age = st.number_input("Donor age", min_value=0, max_value=120, value=30)

    # Recipent gender
    Recipent_gender = st.selectbox("Recipent gender", options=uploaded_data['Recipent gender'].dropna().unique())

    # Donor gender
    Donor_gender = st.selectbox("Donor gender", options=uploaded_data['Donor gender'].dropna().unique())

    # Main diagnosis
    Main_diagnosis = st.selectbox("Main diagnosis", options=uploaded_data['Main diagnosis'].dropna().unique())

    # Diagnosis
    Diagnosis = st.selectbox("Diagnosis", options=uploaded_data['Diagnosis'].dropna().unique())
    
    # Diagnosis
    Subtype = st.selectbox("Subtype of Diagnosis", options=uploaded_data['Subtype'].dropna().unique())
    
    # Stem cell source
    Stem_cell_source = st.selectbox("Stem cell source", options=uploaded_data['Stem cell source'].dropna().unique())

    # Conditioning therapy
    Conditioning_therapy = st.selectbox("Conditioning therapy", options=uploaded_data['Conditioning therapy'].dropna().unique())

    # Conditioning therapy type
    Conditioning_therapy_type = st.selectbox("Conditioning therapy type", options=uploaded_data['Conditioning therapy type'].dropna().unique())

    # Graft-Versus-Host Disease (GVHD) Prophylaxis
    GVHD_Prophylaxis = st.selectbox("Graft-Versus-Host Disease (GVHD) Prophylaxis", options=uploaded_data['Graft-Versus-Host Disease (GVHD) Prophylaxis'].dropna().unique())

    # Recipient blood type
    Recipient_blood_type = st.selectbox("Recipient blood type", options=uploaded_data['Recipient blood type '].dropna().unique())

    # Donor blood type
    Donor_blood_type = st.selectbox("Donor blood type", options=uploaded_data['Recipient blood type '].dropna().unique())
    
    # Blood type mismatch
    if Recipient_blood_type == Donor_blood_type:
        Blood_type_mismatch = False
    else:
        Blood_type_mismatch = True
    
    # HSV infection status
    HSV_infection_status = None # st.checkbox("HSV infection")

    # Engraftment status
    Engraftment_status = None # st.checkbox("Engraftment")

    # Number of donor lymphocyte infusion
    Number_of_donor_lymphocyte_infusion = st.number_input("Number of donor lymphocyte infusion", min_value=0, value=0)

    # Additional variables based on your request

    # Mismatched locus count
    Mismatched_locus_count = st.number_input("Mismatched locus count", min_value=0, max_value=2, value=0)

    # Total Body Irradiation Status
    Total_Body_Irradiation_Status = st.checkbox("Total Body Irradiation")
    
    
    # Gender mismatch
    if Recipent_gender == Donor_gender:
        Gender_mismatch = False # st.checkbox("Gender mismatch")
    else:
        Gender_mismatch = True
        
    # Number of afferes applied to donor
    Number_of_afferes_applied_to_donor = st.number_input("Number of apheresis applied to donor", min_value=0, max_value=5, value=1)

    # The amount of graft given to the recipient
    The_amount_of_graft_given_to_the_recipient = st.number_input("The amount of graft given to the recipient", min_value=0.0, max_value=uploaded_data['The amount of graft given to the recipient'].max(), value=0.0)

    # Mononuclear cells 10E8
    Mononuclear_cells_10E8 = None # st.number_input("Mononuclear cells 10E8", min_value=0.0, value=0.0)

    # CD34-positive cells 10E6
    CD34_positive_cells_10E6 = st.number_input("CD34-positive cells 10E6", min_value=0.0, value=0.0)

    # CD3-positive cells 10E7
    CD3_positive_cells_10E7 = None # st.number_input("CD3-positive cells 10E7", min_value=0.0, value=0.0)

    # CD3-16-56+ 10E7
    CD3_16_56_positive_cells_10E7 = st.number_input("CD3-16-56+ cells 10E7", min_value=0.0, value=0.0)

    # CD3+4+ 10E7
    CD3_4_positive_cells_10E7 = None # st.number_input("CD3+4+ cells 10E7", min_value=0.0, value=0.0)

    # CD3+8+ 10E7
    CD3_8_positive_cells_10E7 = None # st.number_input("CD3+8+ cells 10E7", min_value=0.0, value=0.0)

    # CD19+ 10E7
    CD19_positive_cells_10E7 = None # st.number_input("CD19+ cells 10E7", min_value=0.0, value=0.0)
     
    # Total Nucleated Cells 10E8
    Total_Nucleated_Cells_10E8 = st.number_input("Total Nucleated Cells 10E8", min_value=0.0, max_value=uploaded_data['Total Nucleated Cells 10E8'].max(), value=0.0)
   
    # CMV infection status
    CMV_infection_status = st.checkbox("CMV infection")

    # Presence of hemorrhagic cystitis
    Presence_of_hemorrhagic_cystitis = st.checkbox("Presence of hemorrhagic cystitis")
    if Presence_of_hemorrhagic_cystitis:
        Hemorrhagic_cystitis_grade = st.number_input("Hemorrhagic cystitis grade", min_value=1, max_value=4, value=1)
    else:
        Hemorrhagic_cystitis_grade = 0

    # The day the neutrophil count exceeds 1000
    The_day_the_neutrophil_count_exceeds_1000 = st.number_input("The day the neutrophil count exceeds 1000", min_value=0, max_value=145, value=0)

    # Number of relapse
    Number_of_relapse = st.number_input("Number of relapse", min_value=0, max_value=3, value=0)


    # Additional variables
    Defibrotide_prophylaxis = st.checkbox("Defibrotide prophylaxis")
    # The_day_the_platelet_count_exceeds_20000 = st.number_input("The day the platelet count exceeds 20000", min_value=0, max_value=145, value=0)
    # The_day_the_platelet_count_exceeds_50000 = st.number_input("The day the platelet count exceeds 50000", min_value=0, max_value=145, value=0)

    AGVH_presence = st.checkbox("AGVH presence")

    if AGVH_presence:
        AGVHD_grade = st.number_input("Acute GVHD grade", min_value=1, max_value=4, value=1)
        AGVHD = AGVHD_grade
        if AGVHD_grade == 1:
            AGVH_severity_1vs234 = 0
            AGVH_severity = 0
        elif AGVHD_grade == 2:
            AGVH_severity_1vs234 = 1
            AGVH_severity = 0        
        else:
            AGVH_severity_1vs234 = 1
            AGVH_severity = 1    
        # AGVH_severity_1vs234 = st.checkbox("AGVH severity grade 1 vs. 2,3,4") # st.selectbox("AGVH severity 1 vs 234", options=uploaded_data['AGVH_severity_1vs234'].dropna().unique())
        # AGVH_severity = st.checkbox("AGVH severity grade 1,2 vs. 3,4") # st.selectbox("AGVH severity", options=uploaded_data['AGVH_severity'].dropna().unique())
    else:
        AGVHD = 0
        AGVH_severity_1vs234 = 0
        AGVH_severity = 0
    # AGVHD = st.checkbox("AGVHD")
    # AGVH_severity_1vs234 = st.checkbox("AGVH severity grade 1 vs. 2,3,4") # st.selectbox("AGVH severity 1 vs 234", options=uploaded_data['AGVH_severity_1vs234'].dropna().unique())
    # AGVH_severity = st.checkbox("AGVH severity grade 1,2 vs. 3,4") # st.selectbox("AGVH severity", options=uploaded_data['AGVH_severity'].dropna().unique())
    
    # Collect inputs into a dictionary
    input_data = {
        "HLA_numeric_conformance": HLA_numeric_conformance,
        "Haploidentical transplant": Haploidentical_transplant,
        "Recipent age": Recipent_age,
        "Main diagnosis": Main_diagnosis,
        "Diagnosis": Diagnosis,
        "Subtype" : Subtype,
        "Stem cell source": Stem_cell_source,
        "Conditioning therapy": Conditioning_therapy,
        "Conditioning therapy type": Conditioning_therapy_type,
        "Graft-Versus-Host Disease (GVHD) Prophylaxis": GVHD_Prophylaxis,
        "Recipient blood type ": Recipient_blood_type,
        "Donor blood type": Donor_blood_type,
        "Gender mismatch": Gender_mismatch,
        "HSV infection status": HSV_infection_status,
        "Engraftment status": Engraftment_status,
        "Number of donor lymphocyte infusion": Number_of_donor_lymphocyte_infusion,
        "Mismatched_locus_count": Mismatched_locus_count,
        "Donor age": Donor_age,
        "Total Body Irradiation Status": Total_Body_Irradiation_Status,
        "Blood type mismatch": Blood_type_mismatch,
        "Recipent gender": Recipent_gender,
        "Donor gender": Donor_gender,
        "Number of afferes applied to donor": Number_of_afferes_applied_to_donor,
        "The amount of graft given to the recipient": The_amount_of_graft_given_to_the_recipient,
        "Mononuclear cells 10E8": Mononuclear_cells_10E8,
        "CD34-positive cells 10E6": CD34_positive_cells_10E6,
        "CD3-positive cells 10E7": CD3_positive_cells_10E7,
        "CD3-16-56+ 10E7": CD3_16_56_positive_cells_10E7,
        "CD3+4+ 10E7": CD3_4_positive_cells_10E7,
        "CD3+8+ 10E7": CD3_8_positive_cells_10E7,
        "CD19+ 10E7": CD19_positive_cells_10E7,
        "Total Nucleated Cells 10E8": Total_Nucleated_Cells_10E8,
        "CMV infection status": CMV_infection_status,
        "Presence of hemorrhagic cystitis": Presence_of_hemorrhagic_cystitis,
        "Hemorrhagic cystitis grade": Hemorrhagic_cystitis_grade,
        "The day the neutrophil count exceeds 1000": The_day_the_neutrophil_count_exceeds_1000,
        "Number of relapse": Number_of_relapse,
        # additional
        "Defibrotide prophylaxis": Defibrotide_prophylaxis,
        #"The day the platelet count exceeds 20000": The_day_the_platelet_count_exceeds_20000,
        #"The day the platelet count exceeds 50000": The_day_the_platelet_count_exceeds_50000,
        "AGVHD": AGVHD,
        "AGVH_severity_1vs234": AGVH_severity_1vs234,
        "AGVH_severity": AGVH_severity,
        "AGVH_presence": AGVH_presence,
        }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    #input_df
    
    predict_button_v1 = st.button("Predict")
    
    if predict_button_v1:
        output = predict_target(model, input_df, pred_name)
    #pass



st.write(" ")

