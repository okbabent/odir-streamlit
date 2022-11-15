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

def load_df():
  df = load_dataset.read_odir_data()
  df['Patient Sex'] = df['Patient Sex'].replace(['Female','Male'],[0,1])
  return df

df = load_df()


def split_diag_key_words(pdSeries):
  replacements = ('，', ',')  
  left_diags = pdSeries.apply(lambda c: functools.reduce(lambda s, sep: s.replace(sep, ';'), replacements, c)).str.split(';',  expand=True).stack().reset_index(drop=True)
  return left_diags

left_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[0]])
right_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[1]])

left_diag_keys_str = ','.join(left_diag_keys)
right_diag_keys_str = ','.join(right_diag_keys)


  
  #.str.split('，', expand=True).stack().reset_index(drop=True)
  #right_diags = df.right_diag_key.str.split('，', expand=True).stack().reset_index(drop=True)


#@st.cache(suppress_st_warning=True)
def word_cloud():

  mask_left = np.array(Image.open(utils.get_ressource('assets', 'Leftbw2.jpg')))
  mask_right = np.array(Image.open(utils.get_ressource('assets', 'Rightbw2.jpg')))
  wc_left = WordCloud(background_color="black", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask_left)
  wc_right = WordCloud(background_color="black", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask_right)
  fig, ax = plt.subplots(1,2,figsize=(15,10))
  left=wc_left.generate(left_diag_keys_str) 
  right=wc_right.generate(right_diag_keys_str)
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

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def exploration1():
  sns.set_style('whitegrid',{'grid.linestyle': ':'})

  fig, ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
  fig.subplots_adjust(wspace=0.05)
  #fig.suptitle("\nDistribution de la population en fonction de l'âge : \npopulation totale (gauche) ou répartie selon le sexe (droite)", fontsize=13, fontweight="bold", y=0.02)
  fig.suptitle("\nDistribution de la population en fonction de l'âge", fontsize=13, fontweight="bold", y=0.02)
  ax[0].set_title("Population totale")
  ax[1].set_title("Répartie selon le sexe")
  b=sns.kdeplot(ax=ax[0],x='Patient Age', data=df, legend=False, color='black',linewidth=2.5, fill=True, alpha=.1)
  c=sns.kdeplot(ax=ax[1],x='Patient Age', hue='Patient Sex', data=df, legend=True, palette=['m', 'c'], linewidth=2.5)
  st.pyplot(fig)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def exploration2():
  sns.set_style('whitegrid',{'grid.linestyle': ':'})

  fig, ax = plt.subplots(4,6, sharex=True, sharey=True, figsize=(18,14))
  fig.subplots_adjust(hspace=0.4,wspace=0.06)
  fig.suptitle("\nFigure 4. Distribution de chaque label en fonction de l'âge (gauche) et du sexe (milieu et droite)", fontsize=13, fontweight='bold', y=0.09)

  ax1=sns.kdeplot(ax=ax[0, 0], x='Patient Age', hue='N', data=df, palette=['r', 'g'], linewidth=2)
  ax1.legend(labels=['Normal', 'non-Normal'])
  ax1.text(15, 0.02,'67.4 %', fontsize=12, color='red',  weight='bold')
  ax1.text(10, 0.01,'32.6 %', fontsize=9, color='green', weight='bold')

  ax2=sns.kdeplot(ax=ax[0, 3], x='Patient Age', hue='D', data=df, palette=['r', 'g'], linewidth=2)
  ax2.legend(loc='upper left', labels=['Diabetes', 'non-Diabetes'])
  ax2.text(15, 0.02,'67.8 %', fontsize=9, color='red', weight='bold')
  ax2.text(10, 0.01,'32.2 %', fontsize=9, color='green', weight='bold')

  ax3=sns.kdeplot(ax=ax[1, 0], x='Patient Age', hue='O', data=df, palette=['r', 'g'], linewidth=2)
  ax3.legend(loc='upper left', labels=['Others', 'non-Others'])
  ax3.text(15, 0.02,'72.0 %', fontsize=9, color='red', weight='bold')
  ax3.text(10, 0.01,'28.0 %', fontsize=9, color='green', weight='bold')

  ax4=sns.kdeplot(ax=ax[1, 3], x='Patient Age', hue='G', data=df, palette=['r', 'g'], linewidth=2)
  ax4.legend(loc='upper left', labels=['Glaucoma', 'non-Glaucoma'])
  ax4.text(15, 0.02,'93.9 %', fontsize=9, color='red', weight='bold')
  ax4.text(50, 0.004,'6.1 %', fontsize=9, color='green', weight='bold')

  ax5=sns.kdeplot(ax=ax[2, 0], x='Patient Age', hue='C', data=df, palette=['r', 'g'], linewidth=2)
  ax5.legend(loc='upper left', labels=['Cataract', 'non-Cataract'])
  ax5.text(15, 0.02,'93.9 %', fontsize=9, color='red', weight='bold')
  ax5.text(50, 0.004,'6.1 %', fontsize=9, color='green', weight='bold')

  ax6=sns.kdeplot(ax=ax[2, 3], x='Patient Age', hue='M', data=df, palette=['r', 'g'], linewidth=2)
  ax6.legend(loc='upper left', labels=['Myopia', 'non-Myopia'])
  ax6.text(15, 0.02,'95.0 %', fontsize=9, color='red', weight='bold')
  ax6.text(50, 0.004,'5.0 %', fontsize=9, color='green', weight='bold')

  ax7=sns.kdeplot(ax=ax[3, 0], x='Patient Age', hue='A', data=df, palette=['r', 'g'], linewidth=2)
  ax7.legend(loc='upper left', labels=['AMD', 'non-AMD'])
  ax7.text(10, 0.015,'95.3 %', fontsize=9, color='red', weight='bold')
  ax7.text(50, 0.004,'4.7 %', fontsize=9, color='green', weight='bold')

  ax8=sns.kdeplot(ax=ax[3, 3], x='Patient Age', hue='H', data=df, palette=['r', 'g'], linewidth=2)
  ax8.legend(loc='upper left', labels=['Hypertension', 'non-Hypertension'])
  ax8.text(15, 0.02,'97.1 %', fontsize=9, color='red', weight='bold')
  ax8.text(50, 0.004,'2.9 %', fontsize=9, color='green', weight='bold')

  ax1.set_title("General population \n('Normal' label)", fontdict = {'fontweight':'semibold'})
  ax2.set_title("General population \n('Diabetes' label)", fontdict = {'fontweight':'semibold'})
  ax3.set_title("General population \n('Others' label)", fontdict = {'fontweight':'semibold'})
  ax4.set_title("General population \n('Glaucoma' label)", fontdict = {'fontweight':'semibold'})
  ax5.set_title("General population \n('Cataract' label)", fontdict = {'fontweight':'semibold'})
  ax6.set_title("General population \n('Myopia' label)", fontdict = {'fontweight':'semibold'})
  ax7.set_title("General population \n('AMD' label)", fontdict = {'fontweight':'semibold'})
  ax8.set_title("General population \n('Hypertension' label)", fontdict = {'fontweight':'semibold'})

  ax11=sns.kdeplot(ax=ax[0, 1], x='Patient Age', hue='Patient Sex', data=df[df.N == 1], legend=True, palette=['c', 'm'], linewidth=2, warn_singular=False)
  ax21=sns.kdeplot(ax=ax[0, 4], x='Patient Age', hue='Patient Sex', data=df[df.D == 1], legend=True, palette=['c', 'm'], linewidth=2, warn_singular=False)
  ax31=sns.kdeplot(ax=ax[1, 1], x='Patient Age', hue='Patient Sex', data=df[df.O == 1], legend=True, palette=['c', 'm'], linewidth=2, warn_singular=False)
  ax41=sns.kdeplot(ax=ax[1, 4], x='Patient Age', hue='Patient Sex', data=df[df.G == 1], legend=True, palette=['m', 'c'], linewidth=2, warn_singular=False)
  ax51=sns.kdeplot(ax=ax[2, 1], x='Patient Age', hue='Patient Sex', data=df[df.C == 1], legend=True, palette=['m', 'c'], linewidth=2, warn_singular=False)
  ax61=sns.kdeplot(ax=ax[2, 4], x='Patient Age', hue='Patient Sex', data=df[df.M == 1], legend=True, palette=['m', 'c'], linewidth=2, warn_singular=False)
  ax71=sns.kdeplot(ax=ax[3, 1], x='Patient Age', hue='Patient Sex', data=df[df.A == 1], legend=True, palette=['c', 'm'], linewidth=2, warn_singular=False)
  ax81=sns.kdeplot(ax=ax[3, 4], x='Patient Age', hue='Patient Sex', data=df[df.H == 1], legend=True, palette=['m', 'c'], linewidth=2, warn_singular=False)
                  
  ax11.title.set_text('Normal - per sex')
  ax21.title.set_text('Diabetes - per sex')
  ax31.title.set_text('Others - per sex')
  ax41.title.set_text('Glaucoma - per sex')
  ax51.title.set_text('Cataract - per sex')
  ax61.title.set_text('Myopia - per sex')
  ax71.title.set_text('AMD - per sex')
  ax81.title.set_text('Hypertension - per sex')

  ax10=sns.kdeplot(ax=ax[0, 2], x='Patient Age', hue='Patient Sex', data=df[df.N == 0], legend=True, palette=['m', 'c'], linewidth=2, alpha=0.65, warn_singular=False)
  ax20=sns.kdeplot(ax=ax[0, 5], x='Patient Age', hue='Patient Sex', data=df[df.D == 0], legend=True, palette=['m', 'c'], linewidth=2, alpha=0.65, warn_singular=False)
  ax30=sns.kdeplot(ax=ax[1, 2], x='Patient Age', hue='Patient Sex', data=df[df.O == 0], legend=True, palette=['m', 'c'], linewidth=2, alpha=0.65, warn_singular=False)
  ax40=sns.kdeplot(ax=ax[1, 5], x='Patient Age', hue='Patient Sex', data=df[df.G == 0], legend=True, palette=['m', 'c'], linewidth=2, alpha=0.65, warn_singular=False)
  ax50=sns.kdeplot(ax=ax[2, 2], x='Patient Age', hue='Patient Sex', data=df[df.C == 0], legend=True, palette=['c', 'm'], linewidth=2, alpha=0.65, warn_singular=False)
  ax60=sns.kdeplot(ax=ax[2, 5], x='Patient Age', hue='Patient Sex', data=df[df.M == 0], legend=True, palette=['m', 'c'], linewidth=2, alpha=0.65, warn_singular=False)
  ax70=sns.kdeplot(ax=ax[3, 2], x='Patient Age', hue='Patient Sex', data=df[df.A == 0], legend=True, palette=['m', 'c'], linewidth=2, alpha=0.65, warn_singular=False)
  ax80=sns.kdeplot(ax=ax[3, 5], x='Patient Age', hue='Patient Sex', data=df[df.H == 0], legend=True, palette=['m', 'c'], linewidth=2, alpha=0.65, warn_singular=False)
                  
  ax10.title.set_text('non-Normal - per sex')
  ax20.title.set_text('non-Diabetes - per sex')
  ax30.title.set_text('non-Others - per sex')
  ax40.title.set_text('non-Glaucoma - per sex')
  ax50.title.set_text('non-Cataract - per sex')
  ax60.title.set_text('non-Myopia - per sex')
  ax70.title.set_text('non-AMD - per sex')
  ax80.title.set_text('non-Hypertension - per sex')
  st.pyplot(fig)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def exploration3():
  #df_original = df

  dict1 = {0 : 'Normal',
        1 : 'Diabetes',
        2 : 'Glaucoma',
        3 : 'Cataract',
        4 : 'AMD',
        5 : 'Hypertension',
        6 : 'Myopia',
        7 : 'Others'}
  dict2 = {"C" : 'Cataract',
        "D" : 'Diabetes',
        "G" :'Glaucoma',
        "C" :'Cataract',
        "A" :'AMD',
        "M" :'Myopia',
        "N" :'Normal',
        "H" :'Hypertension',
        "O" :'Others'}
  df_OB = load_dataset.read_csv_data('df_OB.csv', 'Label', dict1) #Dataset Okba
  df_YB = load_dataset.read_csv_data('df_YB.csv', 'Diag', dict2) #Dataset Yannick
  df_TV = load_dataset.read_csv_data('df_TV.csv') #Dataset Thibaut

  st.code(f'Original - nombre de ligne : {df.shape[0]}\n'
  f'OB - nombre de ligne :  {df_OB.shape[0]}\n'
  f'TV - nombre de ligne :  {df_TV.shape[0]}\n'
  f'YB - nombre de ligne :  {df_YB.shape[0]}')

 
  # df_OB['diagnosis'] = df_OB['Label'].replace(dict)
  # df_YB['diagnosis'] = df_YB['Diag'].replace(dict)
  sns.set_style('whitegrid',{'grid.linestyle': ':'})

  fig, ax = plt.subplots(1,3, sharey=True,figsize=(10,3), squeeze=False)
  fig.subplots_adjust(wspace=0.05)

  a=sns.countplot(ax=ax[0,0], x='diagnosis', data=df_OB)
  a.tick_params(axis='x', labelrotation=90)
  b=sns.countplot(ax=ax[0,1], x='diagnosis', data=df_YB)
  b.tick_params(axis='x', labelrotation=90)
  c=sns.countplot(ax=ax[0,2], x='diagnosis', data=df_TV)
  c.tick_params(axis='x',labelrotation=90)
  st.pyplot(fig)

  
def eye_fundus():
  pass


def display():

    ### Create Title
    ui.slide_header("Exploration du dataset", gap=2)
  
    if st.checkbox("Diagnostic keywords"):
      col1, col2 = st.columns(2)
      color = ui.color("blue-green-60")
      with col1:
        st.markdown(f"<h4 style='text-align: center;color: {color}'>Oeil gauche<h4>", unsafe_allow_html=True)
        table = get_keywork_table(df, 0)
        st.table(table)

      with col2:
        st.markdown(f"<h4 style='text-align: center;color: {color}'>Oeil droit<h4>", unsafe_allow_html=True)
        table = get_keywork_table(df, 1)
        st.table(table)

      ui.info(f"Nombre de diagnostics pour les fonds d'oeil gauche : {left_diag_keys.nunique()}")
      ui.info(f"Nombre de diagnostics pour les fonds d'oeil droit : {right_diag_keys.nunique()}")
      # st.markdown("* * *")
      # st.subheader('Nuage de mots clés diagnostiques')
      # word_cloud(df)
      # st.markdown("### > Prédominance des fonds d'oeil normaux à gauche comme à droite")

    if st.checkbox("Nuage de mots clés diagnostics"):
      #st.markdown("* * *")
      #st.markdown(f"<h3 style='text-align: center; color: {color};'La base de données semble ne présente aucunes données absentes ou manquantes</h3>", unsafe_allow_html=True)
      #st.markdown(f"<h3 style='text-align: center; color: {color};'{txt}</h3>", unsafe_allow_html=True)
      word_cloud()
      # color = ui.color("blue-green-60")
      st.markdown('#')
      ui.info("Prédominance des fonds d'oeil normaux à gauche comme à droite")

    if st.checkbox("Exploration - 1"):
      exploration1()
    if st.checkbox("Exploration - 2"):
      exploration2()
    if st.checkbox("Exploration - 3"):
      exploration3()


      



