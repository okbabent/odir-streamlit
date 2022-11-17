import streamlit as st 
from app import ui





MenuChoice = {
  "Métriques" : "A",
  "Matrices de confusions" : "B",
}


def display_choice(menu_choice, args):
    return None





def header():
    return  {'id': "Analyse et Performance des modèles", 'icon': 'graph-up', 'callback': display}


def display():

    ### Create Title
    ui.slide_header("Analyse des performance des modèles", gap=2)
    ui.sub_menus(MenuChoice, display_choice)
   

      
      




