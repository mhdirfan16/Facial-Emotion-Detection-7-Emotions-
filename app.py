import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
import os
import base64

# Load model
model = load_model("emotion_model_7class.h5", compile=False)

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Facial Emotion Detection", layout="wide")

# Function to get base64 string of an image
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ---- CUSTOM CSS ----
# Check if background image exists in the folder
background_image_path = "bg.jpg"  # Change this to your image filename
logo_image_path = "logo.png"  # Your logo file

if os.path.exists(background_image_path):
    # Convert image to base64
    image_base64 = get_base64_of_image(background_image_path)
    background_css = f"""
    /* Create a blurred background layer */
    .blurred-background {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
        background-size: cover;
        filter: blur(10px);
        z-index: -1;
    }}
    
    /* Add a dark overlay for better readability */
    .blurred-background::after {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.4);
        z-index: -1;
    }}
    
    /* Make sure content is above the background */
    .stApp {{
        background: transparent !important;
    }}
    
    .main .block-container {{
        background: transparent;
        position: relative;
        z-index: 1;
    }}
    """
else:
    # Fallback to a gradient background if image doesn't exist
    background_css = """
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    """

# Add logo CSS if logo exists
logo_css = ""
if os.path.exists(logo_image_path):
    logo_base64 = get_base64_of_image(logo_image_path)
    logo_css = f"""
    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }}
    
    .logo {{
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }}
    """

st.markdown(
    f"""
    <style>
    /* Background image with blur effect */
    {background_css}

    /* Logo styling */
    {logo_css}

    /* Glassmorphism container */
    .glass {{
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    /* Title */
    .title {{
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        text-shadow: 2px 2px 8px #000000;
        margin-bottom: 12px;
    }}

    /* Subtitle */
    .subtitle {{
        font-size: 22px;
        font-weight: 400;
        color: #dddddd;
        text-align: center;
        margin-bottom: 35px;
    }}
    
    /* Button styling */
    .stRadio > div {{
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
    }}
    
    /* File uploader styling */
    .stFileUploader > div {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px;
    }}
    </style>
    
    <!-- Create the blurred background layer -->
    <div class="blurred-background"></div>
    """,
    unsafe_allow_html=True
)

# ---- HEADER ----
# Add logo if it exists
if os.path.exists(logo_image_path):
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="data:image/jpg;base64,{logo_base64}" class="logo">
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown('<div class="title">Facial Emotion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze emotions in real-time or from an image</div>', unsafe_allow_html=True)

# ---- LAYOUT ----
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📂 Input Source")

    option = st.radio("Choose input:", ["Webcam", "Upload Image"])

    uploaded_file = None
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Emotion Analysis Summary")

    image = None

    if option == "Upload Image" and uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    elif option == "Webcam":
        img_file_buffer = st.camera_input("Take a photo")
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                roi_gray = np.expand_dims(roi_gray, axis=0)

                preds = model.predict(roi_gray)[0]
                emotion_dict = {emotion_labels[i]: float(preds[i]) for i in range(len(emotion_labels))}
                top_emotion = emotion_labels[np.argmax(preds)]

                # Draw on image
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, top_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                # Show image
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Emotion", use_container_width=True)

                # Chart
                fig = px.bar(
                    x=list(emotion_dict.keys()),
                    y=list(emotion_dict.values()),
                    labels={"x": "Emotion", "y": "Probability"},
                    title="Emotion Probabilities",
                    color=list(emotion_dict.values()),
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.success(f"### 😀 Detected Emotion: **{top_emotion}**")
                break
        else:
            st.error("No face detected.")
    else:
        st.info("Upload an image or take a photo to analyze.")

    st.markdown('</div>', unsafe_allow_html=True)