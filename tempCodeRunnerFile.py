import streamlit as st
from inference_sdk import InferenceHTTPClient
import google.generativeai as genai

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=st.secrets["api_key"]
)
