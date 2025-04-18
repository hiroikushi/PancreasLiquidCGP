import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import pickle
import datetime
from io import BytesIO

path = os.getcwd()
st.header('Prediction of ctDNA detecion by liquid CGP in pancreatic adenocarcinoma patients')
st.markdown('This app predicts the probability of ctDNA detection by liquid CGP tests in patients with pancreatic adenocarcinoma.')
st.subheader('Patient data')

age = st.number_input('Age', min_value=0, max_value=100, value=0, step=1)
sex = st.radio('Sex', ['Woman', 'Man'], horizontal=True)

ps = st.radio('ECOG performance status', ['0', '1', '2', '3', '4'], horizontal=True)

diagdate = st.date_input('Diagnosis date', datetime.date.today())
spedate = st.date_input('Specimen collection date', datetime.date.today())

treatmentline = st.number_input('Current treatment line', min_value=0, max_value=20, value=0, step=1)
response = st.radio('Response', ['PD', 'SD', 'PR', 'CR', 'NE'], horizontal=True)

st.markdown('Metastasis')
lymphmeta = st.toggle('Lymph node', value=False)
lungmeta = st.toggle('Lung', value=False)
pleuralmeta = st.toggle('Pleura', value=False)
livermeta = st.toggle('Liver', value=False)
bonemeta = st.toggle('Bone', value=False)
brainmeta = st.toggle('Brain', value=False)
peritonealmeta = st.toggle('Peritoneum', value=False)
kidneymeta = st.toggle('Kidney', value=False)
adrenalsmeta = st.toggle('Adrenal', value=False)
musclemeta = st.toggle('Muscle', value=False)
softmeta = st.toggle('Soft tissue', value=False)
ovarymeta = st.toggle('Ovary', value=False)



st.subheader('Prediction')
st.markdown('Probability (%) that liquid CGP tests will detect ctDNA:')
button = st.button('Predict')
if button:
    st.write('Predicting...')
    # Data preparation
    if sex == 'Woman':
        sexinput = 0
    else:
        sexinput = 1
       
    ps0 = 1 if ps == '0' else 0
    ps1 = 1 if ps == '1' else 0
    ps2 = 1 if ps == '2' else 0
    ps3 = 1 if ps == '3' else 0
    ps4 = 1 if ps == '4' else 0

    spediag = (spedate - diagdate).days
    if spediag < 0:
        spediag = 0
    
    pd = 1 if response == 'PD' else 0
    sd = 1 if response == 'SD' else 0
    pr = 1 if response == 'PR' else 0
    cr = 1 if response == 'CR' else 0
    ne = 1 if response == 'NE' else 0

    metasite = int(lymphmeta) + int(lungmeta) + int(pleuralmeta) + int(livermeta) + int(bonemeta) + int(brainmeta) + int(peritonealmeta) + int(kidneymeta) + int(adrenalsmeta) + int(musclemeta) + int(softmeta) + int(ovarymeta)

    input = [sexinput, age, spediag, metasite, 
             int(lymphmeta), int(lungmeta), int(pleuralmeta), int(livermeta), int(bonemeta), int(brainmeta), 
             int(peritonealmeta), int(kidneymeta), int(adrenalsmeta), int(musclemeta), int(softmeta), int(ovarymeta), 
             treatmentline, ps0, ps1, ps2, ps3, ps4, cr, ne, pd, pr, sd]
            

    # Prediction
    pred = 0
    fold = 5
    for i in range(fold):
        model = pickle.load(open(f'{path}/pancreasliquidmodel/logistic_250401_shap_{i}.pkl', 'rb'))
        pred += model.predict_proba([input])[:, 1][0] / fold
    pred *= 100

    if pred < 0.01:
        pred = 0.01
    elif pred > 99.99:
        pred = 99.99

    st.write('Prediction done!')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'**<p style="color:red; font-size: 24px; ">Result: {pred:.2f}%</p>**', unsafe_allow_html=True)

    sizes = [pred, 100 - pred]
    explode = (0.1, 0)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie(sizes, explode=explode, startangle=90, counterclock=False, colors=['limegreen', 'lightgrey'])
    ax.axis('equal')
    
    buf = BytesIO() 
    fig.savefig(buf, format="png")
    with col2:
        st.image(buf)

    st.markdown('This result cannot be used for clinical diagnosis. Please consider performing CGP tests at a physician\'s discretion.')
