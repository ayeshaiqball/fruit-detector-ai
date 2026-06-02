import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
from inference_sdk import InferenceHTTPClient
import supervision as sv
import joblib

model = joblib.load("fruit_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Your API Key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Page Setup
st.set_page_config(
    page_title="Fruit AI Project",
    page_icon="🍍",
    layout="wide"
)

# Sidebar Menu
st.sidebar.title("🍍 Fruit AI Project")
st.sidebar.write("By: Ayesha")
page = st.sidebar.selectbox("Choose a Model", [
    "🏠 Home",
    "🤖 Agentic AI",
    "🧠 Machine Learning",
    "🖼️ Deep Learning"
])

# HOME PAGE
if page == "🏠 Home":
    st.title("🍍 Fruit Detection AI Project")
    st.write("Welcome to my AI project!")
    st.write("This website has 3 AI models:")
    st.info("🤖 Agentic AI → Ask questions about fruits")
    st.info("🧠 Machine Learning → Classify fruit from description")
    st.info("🖼️ Deep Learning → Upload image to detect fruit")

# AGENTIC AI PAGE
elif page == "🤖 Agentic AI":
    st.title("🤖 Agentic AI - Fruit Assistant")
    st.write("Ask me anything about fruits!")
    
    user_input = st.text_input("Type your question here:")
    
    if st.button("Ask Gemini AI"):
        if user_input:
            with st.spinner("Thinking..."):
                # Use Flash-Lite here to avoid the "Quota Exceeded" error
                # It is much more stable for free-tier users
                model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
                response = model.generate_content(user_input)
                st.success("Answer:")
                st.write(response.text)
        else:
            st.warning("Please type a question first!")

# MACHINE LEARNING PAGE
elif page == "🧠 Machine Learning":
    st.title("🧠 Machine Learning - Fruit Classifier")
    st.write("Describe a fruit and I will predict what it is!")
    
    user_input = st.text_area("Describe the fruit:")
    if st.button("Predict Fruit"):
      if user_input:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        st.success(f"🍎 Predicted: {prediction}")
      else:
        st.warning("Please enter fruit description")

# DEEP LEARNING PAGE

elif page == "🖼️ Deep Learning":
    st.title("🖼️ Deep Learning - Fruit Image Detector")

    uploaded_file = st.file_uploader(
        "Upload fruit image",
        type=['jpg', 'jpeg', 'png'],
        key="fruit_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Detect Fruit", key="fruit_detect_btn"):
            with st.spinner("Analyzing..."):
                CLIENT = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=st.secrets["api_key"]
                )

                if image.mode == "RGBA":
                    image = image.convert("RGB")

                result = CLIENT.infer(image, model_id="fruit-detector-ej42h/1")
                st.success("Detection complete!")

                detections = sv.Detections.from_inference(result)
                annotated = sv.BoxAnnotator().annotate(
                    scene=np.array(image), detections=detections
                )
                st.image(annotated, caption="Detected Fruits")
                st.json(result)
    else:
        st.write("Upload a fruit image and I will detect it!")