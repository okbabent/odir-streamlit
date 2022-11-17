import streamlit as st
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
from app import load_dataset
from app import ui
from enum import Enum
from typing import List
from streamlit_option_menu import option_menu

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



MenuChoice = {
  # 'A' : "Aperçu des données",
  # 'B' : "Information sur les données",
  # 'C' : "Description des données"
  "Aperçu des données": 'A',
  "Information sur les données": 'B',
  "Description des données" : 'C'
}



def display_choice(menu_choice, args):
  df = args['dataset']
  if menu_choice == 'A':
    def choice_a():
      line_to_plot = st.slider("Selectionner le nombre de lignes à visualiser", min_value=5, max_value = 100)
      st.dataframe(df.head(line_to_plot))
    return choice_a

  if menu_choice == 'B':
    def choice_b():
      df_info = get_dataframe_info(df)
      st.dataframe(df_info.T)
      dtypes = df_info['Dtype'].value_counts()
      st.write(dtypes)
      st.code('Nombre de duplication est égal à [df.duplicated().sum()]  ' + str(df.duplicated().sum()) + '\nLa base de données semble ne présente aucunes données absentes ou manquantes')
    return choice_b

  if menu_choice == 'C':
    def choice_c():
      _,c,_ = st.columns([1,2,1])
      with c:
        df_desc = df.describe().T
        df_desc = df_desc.astype({'count':'int'})
        st.dataframe(df_desc)
    return choice_c

 

 
def header():
  return {'id': "Présentation des données", 'icon': 'book', 'callback': display}

def choice_value(x):
  print('X =',x)
  print('value of x =',x.value)
  return x.value



def display():

    ### Create Title
    ui.slide_header('Présentation des données', gap=(None,None,10), description='Description & information')
  
    df = load_dataset.read_odir_data()
    ui.sub_menus(MenuChoice, display_choice, dataset=df)


    # options=list(MenuChoice.keys())
    # bgc = ui.color("blue-green-70")
    # fgc =  ui.color("blue-green-10")
    # sc =  ui.color("blue-green-50")
    # with st.sidebar:
    #     choose = option_menu("Selectionnez une rubrique", options,
    #                         icons=None,
    #                         menu_icon="list-task", default_index=0,
    #                         styles={
    #         "container": {"padding": "5!important", "background-color": f"{bgc}"},
    #         "icon": {"color": f"{fgc}", "font-size": "20px"}, 
    #         "menu-icon": {"color":"#FF"},
    #         "nav": {"color": "#00FFFF"},
    #         "menu-title": {"font-size": "14px", "font-weight": "bold", "color": "black", "text-align": "left", "margin":"0px"},
    #         "nav-link": {"font-size": "14px", "color": "black", "text-align": "left", "margin":"0px", "--hover-color": "rgb(70, 40, 221)", "--hover-text-color":"#FFFFFF"},
    #         "nav-item": {"color": "#00FF"},
    #         "nav-link-selected": {"background-color": f"{sc}"},
    #     })
    # choice =MenuChoice[choose]
    # fn = display_choice(choice, df)
    # if fn:
    #   fn()
    # selected_rubric = next(item for item in options if item["id"] == choose)
    #selected_rubric = next(item for item in options)
    #print(selected_rubric)
    #selected_rubric['callback']()

    #sub_menus = st.sidebar.radio("Selectionnez une rubrique", options=list(MenuChoice.keys()), format_func=lambda x: x.value)
    #fn = display_choice(sub_menus, df)
    #fn()

   

      
      



