import gradio as gr
from keras.models import load_model
import cv2
import numpy as np


model=load_model('my_model.keras')
blood_group=['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

def prediction_image(image_path):
  image=cv2.resize(image,(128, 128))
  image=np.array(image, dtype='float') / 255.0
  image=np.expand_dims(image, axis=0)
  return image

def model_prediction(image_path):
  image=prediction_image(image_path)
  y_probs=model.predict(image)
  top_3_indices=np.argsort(y_probs[0][-3:][::-1])
  top_three={blood_group[i]: float(y_probs[0][i]) for i in top_3_indices }
  return top_three

Title="Blood Group Prediction"
description="Traditionally, blood group identification requires blood samples and laboratory testing, which can be invasive, time-consuming, and requires specialized equipment. Using fingerprints for blood group identification offers a non-invasive, fast, and potentially portable alternative by leveraging machine learning and image recognition technologies."

demo = gr.Interface(
    fn=model_prediction,
    inputs=gr.Image(label="Upload Image", type="numpy"),
    outputs=gr.Label(num_top_classes=1, label="Predicted Blood Group"),
)

demo.launch()