import streamlit as st
import pandas as pd
from ner_utils import get_named_entities

st.set_page_config(
    page_title="NER using Hugging Face",
    page_icon="🧠",
    layout="wide"
)

st.title("Named Entity Recognition (NER) using Hugging Face Pipeline")
st.write("Enter text below to detect named entities such as Person, Organization, Location, Date, and Miscellaneous.")

sample_text = """Sundar Pichai visited Mumbai on 15 August 2025 for a Google event."""

text = st.text_area("Enter text", value=sample_text, height=180)

if st.button("Extract Entities"):
    with st.spinner("Analyzing text..."):
        entities = get_named_entities(text)

    if entities:
        df = pd.DataFrame(entities)

        st.subheader("Detected Entities")
        st.dataframe(df, use_container_width=True)

        st.subheader("Entity Summary")
        summary = df["label"].value_counts().reset_index()
        summary.columns = ["Entity Type", "Count"]
        st.table(summary)
    else:
        st.warning("No entities detected.")