import streamlit as st
import base64
import os


"""
Utility functions for:
    1. reading data
    2. setting background
    3. writing head and body
"""

@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_working_directory():
  return os.getcwd()

def get_directory(folder_name):
  return os.path.join(get_working_directory(), folder_name);

def get_data_directory():
  return get_directory('data');

def get_assets_directory():
  return get_directory('assets');

def get_ressource(res_folder, res_name):
  return os.path.join(get_directory(res_folder), res_name);