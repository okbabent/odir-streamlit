import streamlit as st 
from app import ui, utils


MenuChoice = {
  "Modélisation - 1" : "A",
  "Modélisation - 2" : "B",
}


def display_choice(menu_choice, args):
    return None


def header():
    return {'id': "Modélisations", 'icon': 'boxes', 'callback': display}

def display():

    ### Create Title
    ui.slide_header("Modélisation des données", gap=2)
    ui.sub_menus(MenuChoice, display_choice)
   

      
      



