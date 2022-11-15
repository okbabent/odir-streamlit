import streamlit as st
import pandas as pd
import requests
import io
from app import utils


def url_to_id(url):
    x = url.split("/")
    return x[5]


@st.cache(suppress_st_warning=True)
def read_odir_data():
  #xls_file = utils.get_ressource('data', 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
  xls_file = 'data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
  return pd.read_excel(xls_file)
  

@st.cache(suppress_st_warning=True)
def read_csv_data(csv_file_name):
  csv_file = 'data/' + csv_file_name
  return pd.read_csv(csv_file)
  

# @st.cache(suppress_st_warning=True)
# def read_data_csv(file_csv_name):
#   # url = 'https://drive.google.com/drive/u/0/folders/1mGluoMZ12iJ1kyDtDNT8nWf441baBitZ/'+file_csv_name
#   #url = 'https://drive.google.com/file/d/1rvfpO2KXS-kd-Nt-78DeO1nYaAs0aPSb/edit'
#   url = 'https://drive.google.com/file/d/1rvfpO2KXS-kd-Nt-78DeO1nYaAs0aPSb/view?usp=sharing'

#   file_id = '1rvfpO2KXS-kd-Nt-78DeO1nYaAs0aPSb'
#   s=requests. get(url).content
#   print(s)
#   csv_content = io.StringIO(s.decode('utf-8'))

#   df = pd.read_csv(csv_content)
#   return df
