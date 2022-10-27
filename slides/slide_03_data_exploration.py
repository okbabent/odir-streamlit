import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
from app import load_dataset
import functools
from nltk.tokenize import PunktSentenceTokenizer
from wordcloud import WordCloud
from PIL import Image
from app import utils

DEFAULT_NUMBER_OF_ROWS = 5
DEFAULT_NUMBER_OF_COLUMNS = 5
DIAGNOSTIC_COL_NAMES = ['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords']

# def split_by_diag_key_word(keywords):
#   replacements = ('，', ',')
#   keywords = functools.reduce(lambda s, sep: s.replace(sep, ';'), replacements, keywords)
#   #diagnostic_keywords = [x.strip() for x in keywords.split(';')]
#   return diagnostic_keywords

def split_diag_key_words(pdSeries):
  replacements = ('，', ',')  
  left_diags = pdSeries.apply(lambda c: functools.reduce(lambda s, sep: s.replace(sep, ';'), replacements, c)).str.split(';',  expand=True).stack().reset_index(drop=True)
  return left_diags
  
  #.str.split('，', expand=True).stack().reset_index(drop=True)
  #right_diags = df.right_diag_key.str.split('，', expand=True).stack().reset_index(drop=True)


#@st.cache(suppress_st_warning=True)
def word_cloud(df):
  left_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[0]]).unique()
  #left_tokens=PunktSentenceTokenizer().tokenize(left_diag_keys)
  right_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[1]]).unique()
  left_keys = ','.join(left_diag_keys)
  right_keys = ','.join(right_diag_keys)
  #right_tokens=PunktSentenceTokenizer().tokenize(right_diag_keys)
  #mask_left = np.array(Image.open("/Users/user/Desktop/Datascientest/Fil_rouge/Leftbw2.jpg"))
  #mask_right = np.array(Image.open("/Users/user/Desktop/Datascientest/Fil_rouge/Rightbw2.jpg"))

  mask_left = np.array(Image.open(utils.get_ressource('assets', 'Leftbw2.png')))
  mask_right = np.array(Image.open(utils.get_ressource('assets', 'Rightbw2.png')))
  wc_left = WordCloud(background_color="white", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask_left)
  wc_right = WordCloud(background_color="white", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask_right)
  fig, ax = plt.subplots(1,2,figsize=(15,10))
  left=wc_left.generate(left_keys) 
  right=wc_right.generate(right_keys)
  ax[0].imshow(wc_left) 
  ax[0].set_title('Left eye fundus')
  ax[1].imshow(wc_right) 
  ax[1].set_title('Right eye fundus')
  ax[0].grid(False)
  ax[1].grid(False)
  ax[0].axis('off')
  ax[1].axis('off')
  st.pyplot(fig)

def set_styles(results):
    table_styles = [
        dict(
            selector="table",
            props=[("font-size", "150%"), ("text-align", "center"), ("color", "red")],
        ),
        dict(selector="caption", props=[("caption-side", "bottom")]),
    ]
    return (
        results.style.set_table_styles(table_styles)
        .set_properties(**{"background-color": "blue", "color": "white"})
        .set_caption("This is a caption")
    )

@st.cache
def _filter_results(results, number_of_rows, number_of_columns) -> pd.DataFrame:
    return results.iloc[0:number_of_rows, 0:number_of_columns]

def filter_results(results, number_of_rows, number_of_columns, style) -> pd.DataFrame:
    filter_table = _filter_results(results, number_of_rows, number_of_columns)
    if style:
        filter_table = set_styles(filter_table)
    return filter_table

def select_number_of_rows_and_columns(results: pd.DataFrame, key: str, select_rows=True, select_columns=True, select_style=True, default_number_of_rows=5, default_number_of_col=5):
    rows = default_number_of_rows
    columns = default_number_of_col
    style= False
    if select_rows:
      rows = st.selectbox(
          "Selectionnez le nombre de ligne à affciher",
          options = [a*100 for a in range(len(results))],
          #options=[5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, len(results)],
          key=key+'_rows',
      )
    if select_columns:
      columns = st.slider(
          "Selectionnez le nombre de colonne à afficher",
          0,
          len(results.columns) - 1,
          default_number_of_col,
          key=key+'_columns',
      )
    if select_style:
      style = st.checkbox("Style dataframe?", False, key=key)
    return rows, columns, style

def missing_values(df):
    flag=0
    for col in df.columns:
            if df[col].isna().sum() > 0:
                flag=1
                missing = df[col].isna().sum()
                portion = (missing / df.shape[0]) * 100
                st.text(f"'{col}': nombre de donnée manquante '{missing}' ==> '{portion:.2f}%'")
    if flag==0:
        st.success("Le dataset ne contient aucune donnée manquante.")


def header():
  return {'id': "Exploration des données", 'icon': 'binoculars', 'callback': display}

def display():

    ### Create Title
    st.title("Exploration du dataset")
    #st.header("Description")
    #st.markdown('La base de données semble ne présenter aucunes données absentes ou manquantes')

     ### Showing code
    #st.text("Lire le dataset: ")
    #with st.echo(): 
      # Normally, you will store all the necessary path and env variables in a .env file
    df = load_dataset.read_odir_data()


    ### Showing the data
    if st.checkbox("Aperçu des données") :
      line_to_plot = st.slider("selectionner le nombre de lignes à visualiser", min_value=3, max_value = 100)
      st.dataframe(df.head(line_to_plot))

    
    if st.checkbox("Description"):
      st.dataframe(df.describe())

    if st.checkbox("Données manquantes ?") : 
      missing_values(df)
    
      #st.dataframe(df.info())
    
    if st.checkbox("Diagnostic keywords 1"):
      eye_side_options=[
            "Oeil gauche",
            "Oeil droit",
        ] 
      diag_side = st.radio('Choisir le côté de l\'oeil',
        options=eye_side_options,
      )
      diag_keys = None
      side_index = eye_side_options.index(diag_side)
      diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[side_index]]).unique()
      data = pd.DataFrame(diag_keys, columns=[DIAGNOSTIC_COL_NAMES[side_index]])
      line_to_plot = st.slider("Selectionezz le nombre de mot-clé à afficher", min_value=1, max_value = data.shape[0], value=15)
      results = data.head(line_to_plot)
      #results = diag_keys[:line_to_plot]
      #number_of_rows, number_of_columns, style = select_number_of_rows_and_columns(results, 'data_frame', True, False, False, 5, 2)
      #print('number_of_rows', number_of_rows)
      #filter_table = filter_results(results, number_of_rows, number_of_columns, style)
      st.table(results)
      #st.dataframe(df.style.highlight_max(axis=0))

    if st.checkbox('Diagnostic keywords 2'):
      word_cloud(df)
      # left_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[0]]).unique()
      # #left_tokens=PunktSentenceTokenizer().tokenize(left_diag_keys)
      # right_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[1]]).unique()
      # left_keys = ','.join(left_diag_keys)
      # right_keys = ','.join(right_diag_keys)
      # #right_tokens=PunktSentenceTokenizer().tokenize(right_diag_keys)
      # #mask_left = np.array(Image.open("/Users/user/Desktop/Datascientest/Fil_rouge/Leftbw2.jpg"))
      # #mask_right = np.array(Image.open("/Users/user/Desktop/Datascientest/Fil_rouge/Rightbw2.jpg"))
    
      # mask = np.array(Image.open(utils.get_ressource('assets', 'mask.png')))
      # wc_left = WordCloud(background_color="white", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask)
      # wc_right = WordCloud(background_color="white", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask)
      # fig, ax = plt.subplots(1,2,figsize=(15,10))
      # left=wc_left.generate(left_keys) 
      # right=wc_right.generate(right_keys)
      # ax[0].imshow(wc_left) 
      # ax[0].set_title('Left eye fundus')
      # ax[1].imshow(wc_right) 
      # ax[1].set_title('Right eye fundus')
      # ax[0].grid(False)
      # ax[1].grid(False)
      # ax[0].axis('off')
      # ax[1].axis('off')
      # st.pyplot(fig)

      
      



