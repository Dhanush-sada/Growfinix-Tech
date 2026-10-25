"""
Streamlit web app for the symptom checker.

Run with:
    streamlit run app.py
"""

import streamlit as st
from predict import Diagnoser

st.set_page_config(page_title="Symptom Checker (Demo)", page_icon="🩺")

st.title("🩺 Symptom Checker — Demo")
st.caption(
    "Educational demo only. This is **not** a real diagnostic tool and should "
    "never replace professional medical advice."
)

@st.cache_resource
def load_diagnoser():
    return Diagnoser()

d = load_diagnoser()

st.subheader("Select your symptoms")
pretty_names = {s: s.replace("_", " ") for s in d.known_symptoms()}
selected = st.multiselect(
    "Type to search and select all symptoms that apply:",
    options=d.known_symptoms(),
    format_func=lambda s: pretty_names[s],
)

top_k = st.slider("Number of conditions to show", min_value=1, max_value=5, value=3)

if st.button("Check possible conditions", type="primary", disabled=(len(selected) == 0)):
    results = d.predict(selected, top_k=top_k)

    st.subheader("Possible conditions")
    for disease, prob in results:
        st.write(f"**{disease}**")
        st.progress(min(prob, 1.0), text=f"{prob*100:.1f}% confidence")

    st.warning(
        "⚠️ This tool uses a simple demo model trained on synthetic data. "
        "It is not a substitute for professional medical evaluation. "
        "If symptoms are severe or worsening, seek medical care."
    )
elif len(selected) == 0:
    st.info("Select one or more symptoms above to see possible conditions.")
