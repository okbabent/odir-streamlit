import streamlit as st
import pandas as pd
from app import utils

@st.cache(suppress_st_warning=True)
def read_odir_data():
  #xls_file = utils.get_ressource('data', 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
  xls_file = 'data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
  return pd.read_excel(xls_file)
  