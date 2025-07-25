import streamlit as st
import logging
from src.generate import generate_text

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

st.set_page_config(page_title="Freeform Text Generator", layout="wide")
st.title("📜 Freeform Text Generator for Content Creators")

st.markdown("""
This app uses deep learning models (T5, GAN with DMK loss, and DCKG) to generate long-form content (750+ words) from 2-3 input context sentences.
""")

# Input Section
st.subheader("📝 Enter 2-3 context sentences")
user_input = st.text_area("Context Input", height=150)

if st.button("Generate Text"):
    if user_input.strip():
        with st.spinner("Generating, please wait..."):
            try:
                result = generate_text(user_input)
                st.success("✅ Generation Complete!")
                st.subheader("📄 Generated Text")
                st.write(result)
            except Exception as e:
                logger.exception("Generation failed")
                st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please provide some context input to generate text.")
