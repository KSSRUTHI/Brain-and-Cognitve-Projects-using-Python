import streamlit as st
import pandas as pd
import numpy as np
from backend import generate_synthetic_eeg, filter_band, compute_band_power, save_session_data
import sounddevice as sd

if "eeg_data" not in st.session_state:
    st.session_state.eeg_data = []


st.title("Brainwave-Based Meditation App - (Simulated)")
st.sidebar.title("Settings")
alpha_threshold = st.sidebar.slider("Alpha Relaxation Threshold", 10, 100, 50)
session_filename = st.sidebar.text_input("Session Data Filename", "synthetic_eeg_session.csv")


if st.button("Start Meditation"):
    st.write("Starting meditation with simulated EEG data...")
    for _ in range(100):  
        eeg_data = generate_synthetic_eeg()
        alpha_data = filter_band(eeg_data, 256, "alpha")
        beta_data = filter_band(eeg_data, 256, "beta")
        theta_data = filter_band(eeg_data, 256, "theta")

        alpha_power = compute_band_power(alpha_data)
        beta_power = compute_band_power(beta_data)
        theta_power = compute_band_power(theta_data)

        st.session_state.eeg_data.append({
            "alpha": alpha_power,
            "beta": beta_power,
            "theta": theta_power
        })

    
        st.line_chart(pd.DataFrame(st.session_state.eeg_data))

        if alpha_power > alpha_threshold:
            st.success("Relaxation achieved! Playing calming sound...")
            sd.play(np.sin(2 * np.pi * np.arange(44100) * 440.0 / 44100), 44100)
        else:
            st.warning("Focus on your breathing to relax.")


if st.button("Stop Meditation"):
    st.write("Meditation session ended.")

if st.button("Save Session Data"):
    filepath = save_session_data(st.session_state.eeg_data, session_filename)
    st.success(f"Session data saved to {filepath}")


st.sidebar.subheader("Additional Visualizations")
if st.sidebar.checkbox("Show Beta Band Activity"):
    beta_df = pd.DataFrame([d["beta"] for d in st.session_state.eeg_data], columns=["Beta Power"])
    st.sidebar.line_chart(beta_df)

if st.sidebar.checkbox("Show Theta Band Activity"):
    theta_df = pd.DataFrame([d["theta"] for d in st.session_state.eeg_data], columns=["Theta Power"])
    st.sidebar.line_chart(theta_df)
