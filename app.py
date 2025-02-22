# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Titre de l'application
st.title("Prédiction de Réservation Hôtelière 🏨")


# Charger le modèle XGBoost
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("xgboost_model.json")  # Charger le modèle XGBoost
    return model

# Charger le scaler
@st.cache_resource
def load_scaler():
    scaler = joblib.load("scaler.pkl")  # Charger le scaler
    return scaler

# Charger le modèle et le scaler
model = load_model()
scaler = load_scaler()

# Formulaire pour saisir les données
st.sidebar.header("Saisissez les informations de la réservation")

# Fonction pour saisir les données
def user_input_features():
    no_of_adults = st.sidebar.number_input("Nombre d'adultes", min_value=0, max_value=10, value=2)
    no_of_children = st.sidebar.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.sidebar.number_input("Nombre de nuits le week-end", min_value=0, max_value=10, value=1)
    no_of_week_nights = st.sidebar.number_input("Nombre de nuits en semaine", min_value=0, max_value=10, value=2)
    lead_time = st.sidebar.number_input("Délai de réservation (en jours)", min_value=0, max_value=365, value=30)
    repeated_guest = st.sidebar.selectbox("Client répété ?", options=[-1, 1], format_func=lambda x: "Non" if x == -1 else "Oui")
    no_of_previous_cancellations = st.sidebar.number_input("Nombre d'annulations précédentes", min_value=0, max_value=10, value=0)
    no_of_previous_bookings_not_canceled = st.sidebar.number_input("Nombre de réservations non annulées", min_value=0, max_value=10, value=0)
    no_of_special_requests = st.sidebar.number_input("Nombre de demandes spéciales", min_value=0, max_value=10, value=0)

    # Créer un DataFrame avec les données saisies
    data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'lead_time': lead_time,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'no_of_special_requests': no_of_special_requests,
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Saisie des données
input_df = user_input_features()

# Afficher les données saisies
st.subheader("Données saisies")
st.write(input_df)

# Prétraitement des données
input_scaled = scaler.transform(input_df)

# Prédiction
if st.sidebar.button("Prédire"):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader("Résultat de la prédiction")
    if prediction[0] == 1:
        st.success("La réservation ne sera **pas annulée**.")
    else:
        st.error("La réservation sera **annulée**.")

    st.subheader("Probabilités")
    st.write(f"Probabilité d'annulation : {prediction_proba[0][0]:.2f}")
    st.write(f"Probabilité de non-annulation : {prediction_proba[0][1]:.2f}")