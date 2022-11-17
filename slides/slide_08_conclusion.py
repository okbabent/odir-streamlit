import streamlit as st 
from app import ui




MenuChoice = {
  "Conclusion" : "A",
  "Perspectives" : "B",
  "La Fin" : "C"
}


def display_choice(menu_choice, args):
    if menu_choice == 'A':
        pass
    elif menu_choice == 'B':
        pass
    elif menu_choice == 'C':
        st.balloons()

def header():
    return {'id': "Conclusion & Perspectives", 'icon': 'eyeglasses', 'callback': display}


def display():
    ### Create Title
     ### Create Title
    ui.slide_header("Conclusion & Perspectives", gap=2)
    ui.sub_menus(MenuChoice, display_choice)
   
    
   

      
      



