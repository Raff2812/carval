import streamlit as st
import joblib
import pandas as pd

# Ora puoi caricare i file senza cambiare directory
model_path = "random_forest_regressor_model.pkl"
transformers_path = "pipeline_regressor_transformers.pkl"

@st.cache_resource
def load_resources():
    model = joblib.load(model_path)
    transformers = joblib.load(transformers_path)
    known_makes = transformers['preparator'].encoder.known_makes
    marca_modelli_dict = transformers['preparator'].encoder.marca_modelli_dict  # Dizionario marca -> modelli
    return model, transformers, known_makes, marca_modelli_dict

model, transformers, known_makes, marca_modelli_dict = load_resources()

# Titolo dell'applicazione
st.title("Stima del prezzo di un'auto")

# Selezione marca
marca = st.selectbox("Inserisci la marca", list(known_makes))

# Ottieni i modelli associati alla marca selezionata
modelli_disponibili = marca_modelli_dict.get(marca, [])  # Lista vuota se la marca non è nel dizionario

# Usa st.session_state per evitare reset del modello selezionato quando cambia la marca
if "modello" not in st.session_state or st.session_state.marca != marca:
    st.session_state.modello = modelli_disponibili[0] if modelli_disponibili else ""
    st.session_state.marca = marca  # Aggiorna la marca selezionata

# Selezione modello basata sulla marca scelta
modello = st.selectbox("Inserisci il modello:", modelli_disponibili, index=0 if modelli_disponibili else None, key="modello")

# Altri campi di input
anno_produzione = st.number_input("Inserisci l'anno di produzione:", min_value=1900, max_value=2015, value=2010, step=1)
allestimento = st.selectbox("Inserisci l'allestimento:", ['base', 'sport', 'luxury', 'special edition', 'touring',  'other'])
carrozzeria = st.selectbox("Inserisci la carrozzeria:", ['sedan', 'suv', 'hatchback', 'coupé', 'cabriolet', 'station wagon', 'pickup', 'other'])

# ⭐ Condizione dell'auto con stelle interattive
st.write("Inserisci la condizione dell'auto:")

if "condizione" not in st.session_state:
    st.session_state.condizione = 3  # Default

cols = st.columns(5)
for i in range(5):
    with cols[i]:
        star_label = "⭐" if i < st.session_state.condizione else "☆"
        if st.button(star_label, key=f"star_{i}"):
            st.session_state.condizione = i + 1  # Aggiorna la condizione

chilometraggio = st.number_input("Inserisci il chilometraggio:", min_value=0, value=0, step=500)
colore = st.text_input("Inserisci il colore esterno:")
interni = st.text_input("Inserisci il colore degli interni:")

if st.button("Stima il prezzo"):
    input_data = {
        'marca': [marca],
        'modello': [modello],
        'trasmissione': 'automatic',
        'anno produzione': [int(anno_produzione)],
        'allestimento': [allestimento],
        'carrozzeria': [carrozzeria],
        'condizione': [int(st.session_state.condizione)],
        'chilometraggio': [int(chilometraggio)],
        'colorazione': [colore],
        'colore interni': [interni]
    }

    df_input = pd.DataFrame(input_data)
    df_input = transformers['preparator'].transform_test(df_input)

    prezzo_predetto = model.predict(df_input)[0]

    st.success(f"Il prezzo stimato è: {prezzo_predetto:.2f}€")
