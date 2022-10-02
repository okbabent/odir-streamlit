# Core Pkg
import streamlit as st

# Custom modules
from streamlit_odir_intro import streamlit_odir_intro
from streamlit_odir_data_processing import streamlit_data_preprocessing
from streamlit_odir_data_exploration import streamlit_data_exploration

from streamit_experiments import streamlit_experiments
# from demo_stream_titanic import demo_streamlit # Basic ML web app with stremlit

def main():

    # List of pages
    liste_menu = ["Introduction", "Exploration des données", "Traitement des données", "Experiments"]
    list_callbacks = [streamlit_odir_intro, streamlit_data_exploration, streamlit_data_preprocessing, streamlit_experiments]

    # Sidebar
    menu = st.sidebar.selectbox("Présentation", liste_menu)

    # Page navigation
    menu_index = liste_menu.index(menu)
    list_callbacks[menu_index]()
    # if menu == liste_menu[0]:
    #     bases_streamlit()
    # else:
    #     demo_streamlit()


if __name__ == '__main__':
    main()