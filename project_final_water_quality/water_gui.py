import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import joblib
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title='water quality prediction',
    page_icon=':gem:',
    initial_sidebar_state='collapsed'  # Collapsed sidebar
)

def load_lottie(url):  # test url if you want to use your own lottie file 'valid url' or 'invalid
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

regmodel = joblib.load(open("project_model", "rb"))
svmmodel=joblib.load(open("svm_model","rb"))
dtmodel=joblib.load(open("decision_tree_model","rb"))
rfmodel=joblib.load(open("random_forest_model","rb"))

def predictlog(ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    features = np.array([ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]).reshape(1, -1)
    prediction = regmodel.predict(features)
    return prediction
def predictsvm(ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    features = np.array([ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]).reshape(1, -1)
    prediction = svmmodel.predict(features)
    return prediction
def predictdt(ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    features = np.array([ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]).reshape(1, -1)
    prediction = dtmodel.predict(features)
    return prediction
def predictrf(ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    features = np.array([ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]).reshape(1, -1)
    prediction = rfmodel.predict(features)
    return prediction

with st.sidebar:
    choose = option_menu(None, ["Home"],
                         icons=['house'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "#fafafa"},
                             "icon": {"color": "#E0E0EF", "font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                             "nav-link-selected": {"background-color": "#02ab21"},
                         }
                         )

if choose == 'Home':
    st.write('# water quality prediction')
    st.write("choose a model")
    choicerg=st.checkbox("logistic regresion")
    choicesvm=st.checkbox("svm") 
    choicedt=st.checkbox("decision tree")
    choicerf=st.checkbox("random forest")
    st.subheader('enter details to predict')
    ph = st.number_input("enter PH value",min_value=0.0,  step=0.01, format="%.2f")
    Hardness = st.number_input("enter hardness value", min_value=0.00,step=0.01)
    Solids = st.number_input("enter solids value", min_value=0.00,step=0.01)
    Sulfate = st.number_input("enter sulfate value", min_value=0.00,step=0.01)
    Chloramines = st.number_input("enter chloramines value", min_value=0.00,step=0.01)
    Conductivity = st.number_input("enter conductivity value", min_value=0.00,step=0.01)
    Organic_carbon = st.number_input("enter organic_carbon value", min_value=0.0,step=0.01)
    Trihalomethanes = st.number_input("enter Trihalomethanes value", min_value=0.00,step=0.01)
    Turbidity = st.number_input("enter Turbidity value", min_value=0.00,step=0.01)
  
    
    samplerg = predictlog(ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
    samplesvm=predictsvm(ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
    sampledt=predictdt(ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
    samplerf = predictrf(ph, Hardness, Solids, Sulfate, Chloramines, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
  
    btn=st.button("submit")
   #checkbox for regression
    if btn==True:
        if choicerg ==True:
          if samplerg == False:
              st.write(" regression: The water is unfit for consumption.")
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\not_safe_to_drink.png')
          elif samplerg == True:
              st.write("regression: this water is fit for consumption")
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\safe_to_drinkall.gif')
              st.balloons()
          st.image('D:\\college\\second year\\second term\\Artificial intelligence\\project_final_water_quality\\output.png')  
          st.text('accuracy logistic regression: 0.510078')
         #checkbox for svm   
        if choicesvm ==True:
           if samplesvm == False:
              st.write(" svm: The water is unfit for consumption.")
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\not_safe_to_drink.png')
           elif samplesvm == True:
              st.write("svm: this water is fit for consumption")
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\safe_to_drinkall.gif')
              st.balloons()   
           st.image('D:\\college\\second year\\second term\\Artificial intelligence\\project_final_water_quality\\svm.png') 
           st.text('accuracy svm:0.770543')     
       #checkbox for descion tree 
        if choicedt ==True:
          if sampledt == False:
              st.write(" decision tree: The water is unfit for consumption.")
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\not_safe_to_drink.png')
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\project_final_water_quality\\desciontree.png')  
          elif sampledt == True:
              st.write(" decision tree: this water is fit for consumption")
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\safe_to_drinkall.gif')
              st.balloons() 
          st.image('D:\\college\\second year\\second term\\Artificial intelligence\\project_final_water_quality\\desciontree.png')  
          st.text('accuracy decision tree:0.8')
        if choicerf ==True:
           if samplerf == False:
              st.write("random forrest: The water is unfit for consumption.")
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\not_safe_to_drink.png')
           elif samplerf == True:
              st.write(" random forest: this water is fit for consumption")
              st.image('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\safe_to_drinkall.gif')
              st.balloons()  
           st.image('D:\\college\\second year\\second term\\Artificial intelligence\\project_final_water_quality\\randomforest.png')       
           st.text('accuracy random forest:0.803101');

data = pd.read_csv('D:\\college\\second year\\second term\\Artificial intelligence\\ai_project_final\\water_potability.csv')
df = pd.DataFrame(data)

