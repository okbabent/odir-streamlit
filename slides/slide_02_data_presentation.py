import streamlit as st
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
from app import load_dataset


def header():
  return {'id': "Présentation des données", 'icon': 'book', 'callback': display}

def display():

    ### Create Title
    st.title("Présentation des données")

        ### Create Title
    st.title("Description & information sur la base de données")
    st.header("Description")
    st.markdown('La base de données semble ne présenter aucunes données absentes ou manquantes')

     ### Showing code
    #st.text("Lire le dataset: ")
    #with st.echo(): 
      # Normally, you will store all the necessary path and env variables in a .env file
    df = load_dataset.read_odir_data()


    ### Showing the data
    if st.checkbox("Aperçu des données") :
      line_to_plot = st.slider("selectionner le nombre de lignes à visualiser", min_value=3, max_value = 100)
      st.dataframe(df.head(line_to_plot))

    if st.checkbox("Données manquantes ?") : 
      st.dataframe(df.isna().sum())
      #st.dataframe(df.info())

    if st.checkbox("Description"):
      st.dataframe(df.describe())

   

      
      



