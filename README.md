# Facial Emotion Detection (7 Emotions) 🤖

This is a Machine Learning project designed to detect and classify human emotions from facial expressions. The system recognizes 7 emotions:
👉 Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The project combines deep learning with computer vision to analyze facial features and predict emotions accurately. It comes with a Streamlit web interface that allows users to test the model easily, either through real-time webcam input or by uploading images.

## 🚀 Features

✅ Real-Time Emotion Detection – Uses your webcam to analyze expressions on the fly.

✅ Image Upload Support – Upload an image and get instant emotion classification.

✅ Interactive Visualization – Displays probability distribution for all emotions using Plotly.

✅ Clean & Modern UI – Streamlit interface with blurred background and styled components.

✅ Face Detection – Powered by OpenCV’s Haar Cascade for robust face localization.


## 🛠️ Tech Stack

Python – Core programming language.

TensorFlow / Keras – Deep learning framework used to train the model.

OpenCV – For face detection and preprocessing images.

Streamlit – To deploy the web app with an interactive user interface.

Plotly – For displaying probability charts.

## 📊 Model Information

Model: Trained CNN model (emotion_model_7class.h5).

Dataset: Standard facial expression dataset (7 emotion categories).

Input: Grayscale face images (48x48).

Output: Softmax probabilities across 7 classes.

Performance: Achieved ~79% accuracy with 20 epochs.

The model learns to capture subtle facial features such as eyebrows, mouth curvature, and eye openness to identify emotions.

## 🌟 Future Enhancements

🔹 Improve accuracy with larger datasets (FER+ / AffectNet).

🔹 Optimize for faster inference on real-time webcam input.

🔹 Multi-face detection in a single frame.

🔹 Cloud deployment on Streamlit Cloud / Heroku / AWS for public access.

🔹 Add support for continuous video stream processing.


## 🎯 Use Cases

Human–Computer Interaction (HCI).

Mental health monitoring.

Sentiment analysis in classrooms or workplaces.

Entertainment & gaming (detecting player emotions).

Customer feedback systems in businesses.
