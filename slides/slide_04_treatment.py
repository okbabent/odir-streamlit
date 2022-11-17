import streamlit as st 
from app import ui, utils
from enum import Enum
from typing import List





MenuChoice = {
  "Traitement des mots clés" : "A",
  "Traitement des images" : "B",
}

def display_choice(menu_choice, args):
    return None

def header():
    return {'id': "Traitement des données", 'icon': 'bar-chart', 'callback': display}

def display():
    ### Create Title
    ui.slide_header("Traitement des données", gap=2)
    ui.sub_menus(MenuChoice, display_choice)
   
   

      
      



