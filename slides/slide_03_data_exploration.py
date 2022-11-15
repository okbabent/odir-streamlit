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
from app import ui

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
  left_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[0]])
  right_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[1]])
  left_diag_keys = ','.join(left_diag_keys)
  right_diag_keys = ','.join(right_diag_keys)

  # left_eye = pd.DataFrame(data=df['Left-Diagnostic Keywords'].astype(str))
  # left_eye_keys = left_eye['Left-Diagnostic Keywords'].str.cat(sep=', ')
  # right_eye=pd.DataFrame(data=df['Right-Diagnostic Keywords'].astype(str))
  # right_eye=right_eye['Right-Diagnostic Keywords'].str.cat(sep=', ')

  mask_left = np.array(Image.open(utils.get_ressource('assets', 'Leftbw2.jpg')))
  mask_right = np.array(Image.open(utils.get_ressource('assets', 'Rightbw2.jpg')))
  wc_left = WordCloud(background_color="black", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask_left)
  wc_right = WordCloud(background_color="black", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask_right)
  fig, ax = plt.subplots(1,2,figsize=(15,10))
  left=wc_left.generate(left_diag_keys) 
  right=wc_right.generate(right_diag_keys)
  fig.set_facecolor('black')
  fig.tight_layout(pad=2.0)
  alpha = 1
  fig.set_alpha(alpha)
  ax[1].set_facecolor('black')
  ax[0].set_facecolor('black')
  ax[1].set_alpha(alpha)
  ax[0].set_alpha(alpha)
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

def get_keywork_table(df, side):
    diag_keys = None
    side_index = side
    diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[side_index]]).unique()
    data = pd.DataFrame(diag_keys, columns=[DIAGNOSTIC_COL_NAMES[side_index]])
    line_to_plot = st.slider("Selectionez le nombre de mot-clé à afficher", min_value=1, max_value = data.shape[0], value=15)
    table = data.head(line_to_plot)
    return table

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

    if st.checkbox("Diagnostic keywords"):
      col1, col2 = st.columns(2)
      with col1:
        st.subheader('Oeil gauche')
        table = get_keywork_table(df, 0)
        st.table(table)

      with col2:
        st.subheader('Oeil droit')
        table = get_keywork_table(df, 1)
        st.table(table)
      # st.markdown("* * *")
      # st.subheader('Nuage de mots clés diagnostiques')
      # word_cloud(df)
      # st.markdown("### > Prédominance des fonds d'oeil normaux à gauche comme à droite")

    if st.checkbox("Nuage de mots"):
      #st.markdown("* * *")
      txt = ui.title_label('Nuage de mots clés diagnostics')
      color = "#FFFFFF"
      #st.markdown(f"<h3 style='text-align: center; color: {color};'La base de données semble ne présente aucunes données absentes ou manquantes</h3>", unsafe_allow_html=True)
      st.markdown(f"<h3 style='text-align: center; color: {color};'{txt}</h3>", unsafe_allow_html=True)
      word_cloud(df)
      #st.markdown("### Prédominance des fonds d'oeil normaux à gauche comme à droite")
      color = ui.color("blue-green-60")
      st.write(f"<h3 style='text-align: center; color: {color};'Prédominance des fonds d'oeil normaux à gauche comme à droite</h3>", unsafe_allow_html=True)


      



