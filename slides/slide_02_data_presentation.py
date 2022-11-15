import streamlit as st
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
from app import load_dataset
from app import ui

def get_dataframe_info(df):
    """
    input
       df -> DataFrame
    output
       df_null_counts -> DataFrame Info (sorted)
    """

    df_info = pd.DataFrame({
      "Column": df.columns, 
      "Non-Null count": len(df)-df.isnull().sum().values, 
      "Null count": df.isnull().sum().values, 
      "Dtype": df.dtypes.astype(str).values})

   
    

    return df_info


 
def header():
  return {'id': "Présentation des données", 'icon': 'book', 'callback': display}

def display():

    ### Create Title
    st.title("Présentation des données")

        ### Create Title
    st.title("Description & information sur la base de données")
    st.header("Description")
   

     ### Showing code
    #st.text("Lire le dataset: ")
    #with st.echo(): 
      # Normally, you will store all the necessary path and env variables in a .env file
    df = load_dataset.read_odir_data()
    #df = load_dataset.read_csv_data('df_TV.csv')


    ### Showing the data
    if st.checkbox("Aperçu des données") :
      line_to_plot = st.slider("selectionner le nombre de lignes à visualiser", min_value=5, max_value = 100)
      st.dataframe(df.head(line_to_plot))

    if st.checkbox("Information sur les données") : 
      #st.dataframe(df.isna().sum())
      df_info = get_dataframe_info(df)
      st.dataframe(df_info.T)
      dtypes = df_info['Dtype'].value_counts()
      st.write(dtypes)
      #st.markdown('#### *La base de données semble ne présenter aucunes données absentes ou manquantes*')
     
      color = ui.color("blue-green-60")
      st.markdown(f"<h3 style='text-align: center; color: {color};'La base de données semble ne présente aucunes données absentes ou manquantes</h3>", unsafe_allow_html=True)
      st.code('Nombre de duplication est égal à ' + str(df.duplicated().sum()))
      #st.write('Nombre de duplication est égal à ' + str(df.duplicated().sum()))
      

    if st.checkbox("Description des données"):
      df_desc = df.describe().T
      df_desc = df_desc.astype({'count':'int'})
      st.dataframe(df_desc)

   

      
      



