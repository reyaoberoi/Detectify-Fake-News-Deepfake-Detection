import gradio as gr
import joblib
import numpy as np
from PIL import Image
import os

# Deep learning model loader
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

# ---- Load Models ----
try:
    fake_news_model = joblib.load("Final models/fake_news_model.pkl")
    vectorizer = joblib.load("Final models/vectorizer.pkl")
    print("Fake news model and vectorizer loaded.")
except Exception as e:
    print("Error loading fake news model/vectorizer:", str(e))
    fake_news_model, vectorizer = None, None

try:
    deepfake_model = load_model("Final models/deepfake_detector.h5")
    print("Deepfake model loaded.")
except Exception as e:
    print("Error loading deepfake model:", str(e))
    deepfake_model = None

#Fake News Detection
def analyze_text(input_text):
    try:
        if not fake_news_model or not vectorizer:
            raise ValueError("Fake news model or vectorizer not loaded.")

        vectorized = vectorizer.transform([input_text])
        prediction = fake_news_model.predict(vectorized)
        label = "Real" if int(prediction[0]) == 1 else "Fake"
        explanation = f"The article is classified as {label.lower()} based on its language patterns."
        return label, explanation

    except Exception as e:
        return "Error", f" {str(e)}"

#Deepfake Detection
def analyze_image(image: Image.Image):
    try:
        if not deepfake_model:
            raise ValueError("Deepfake model not loaded.")

        # Preprocess image: resize and scale
        image = image.convert("RGB")
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = deepfake_model.predict(img_array)[0][0]
        label = "Fake" if prediction >= 0.5 else "Real"
        confidence = round(float(prediction) * 100, 2) if prediction >= 0.5 else round((1 - float(prediction)) * 100, 2)
        explanation = f"The image is classified as {label.lower()} with {confidence}% confidence."

        return label, explanation

    except Exception as e:
        return "Error", f" {str(e)}"

# ---- Gradio UI ----
with gr.Blocks(css=".gr-button {background-color: #6366F1; color: white;}") as demo:
    gr.Markdown(
        """
        # 🕵️‍♀️ Detectify - Fake News & Deepfake Detection  
        Analyze suspicious news articles and images for fake content.
        """, elem_id="title")

    with gr.Tabs():
        # --- Fake News Detection Tab ---
        with gr.Tab("Fake News Detection"):
            with gr.Column():
                text_input = gr.Textbox(
                    placeholder="Paste news article or headline here",
                    label="News Text"
                )
                analyze_button = gr.Button("🔍 Analyze Text")
                prediction_output = gr.Textbox(label="Prediction")
                explanation_output = gr.Textbox(label="Explanation")

                analyze_button.click(
                    fn=analyze_text,
                    inputs=text_input,
                    outputs=[prediction_output, explanation_output]
                )

        # --- Deepfake Detection Tab ---
        with gr.Tab("Deepfake Detection"):
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image (face frame)")
                analyze_image_button = gr.Button("🖼️ Analyze Image")
                image_pred_output = gr.Textbox(label="Prediction")
                image_explanation_output = gr.Textbox(label="Confidence")

                analyze_image_button.click(
                    fn=analyze_image,
                    inputs=image_input,
                    outputs=[image_pred_output, image_explanation_output]
                )

# Run app
if __name__ == "__main__":
    demo.launch()
