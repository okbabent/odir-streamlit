import streamlit as st 
from app import ui




MenuChoice = {
  "Conclusion" : "A",
  "Perspectives" : "B",
  "Q & A" : "C"
}

# Résultats :
# - Bon apprentissage des modèles avec de bons résultats (F1-score >0.96)
# - Matrice de confusion montre mauvaise distinction entre les classes Myopie et Glaucome

# Discussion : 
# - Intérêt d’une segmentation d’image avec l’appui d’experts en analyse d’image et d’experts médicaux afin de déterminer les zones d’intérêt à analyser pour la classification
# - Intérêt d’une validation de la représentativité de la dataset par des experts médicaux (distribution des populations, représentation de chaque classe …) pour évaluer une potentielle généralisation de nos modèles.
# - Intérêt d’une validation sur les images tests ou d’autres images pour valider nos modèles.


def display_choice(menu_choice, args):
    if menu_choice == 'A':
        def choice_a():
            _ , c= st.columns([1,8])
            with c:
                ui.add_vgap(20)
                st.markdown(
                    """
                
                    - ## Plusieurs architectures de modèles testés VGG16, VGG19, RESNET50, INCEPTION, XCEPTION
                    - ## Plusieurs tentives pour le choix des hyperparamètres
                    - ## Bon apprentissage des modèles avec de bons résultats (F1-score >0.96)
                    - ## Matrices de confusions montrent de mauvaises distinctions entre les classes Myopie et Glaucome

                    """
                )
        return choice_a
    elif menu_choice == 'B':
        def choice_b():
            _ , c= st.columns([1,8])
            with c:
                ui.add_vgap(20)
                st.markdown(
                    """
                    - ## Intérêt d’une segmentation d’image avec l’appui d’experts en analyse d’image et d’experts médicaux afin de déterminer les zones d’intérêt à analyser pour la classification
                    - ## Intérêt d’une validation de la représentativité de la dataset par des experts médicaux (distribution des populations, représentation de chaque classe …) pour évaluer une potentielle généralisation de nos modèles.
                    - ## Intérêt d’une validation sur les images tests ou d’autres images pour valider nos modèles.
                    - ## Il est indispensabale d'avoir accès à des ressources avec GPU pour traiter ce type de sujets...
                    """
                )
        return choice_b
        pass
    elif menu_choice == 'C':
        _ , c, _= st.columns([1,2,1])
        with c:
            ui.add_vgap(15)
            st.image('assets/Q&A-as-a-logo.png')
        st.balloons()

def header():
    return {'id': "Conclusion & Perspectives", 'icon': 'eyeglasses', 'callback': display}


def display():
    ### Create Title
     ### Create Title
    ui.slide_header("Conclusion & Perspectives", gap=2)
    ui.sub_menus(MenuChoice, display_choice)
   
    
   

      
      



