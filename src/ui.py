import streamlit as st
import requests
import os
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🤖")

st.title("🤖 BERT Sentiment Analyzer")
st.markdown("Enter text below to analyze its sentiment (Positive/Negative).")

text_input = st.text_area("Enter text:", height=150)

if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                payload = {"text": text_input}
                response = requests.post(f"{API_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]
                    
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Sentiment", sentiment.capitalize())
                        
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                    if sentiment == "positive":
                        st.balloons()
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is it running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.caption("Powered by FastAPI & BERT")
