import streamlit as st
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import tempfile

# Load models
detection_model = YOLO("detection.pt")
classification_model = YOLO("classification.pt")

# Carbon map
carbon_map = {
    "neem": 26,
    "banyan": 40,
    "coconut": 35,
    "amla": 18.5,
    "mango": 30,
    "pine": 22,
    "others": 20
}

name_map = {
    "pine tree": "pine",
    "mango tree": "mango",
    "neem tree": "neem",
    "banyan tree": "banyan",
    "amla tree": "amla",
    "coconut tree": "coconut"
}


# Streamlit UI
st.title("🌳 Tree Detection + Classification + Carbon Estimation")

uploaded_file = st.file_uploader("Upload an image of a tree area", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    image = cv2.imread(temp_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detection
    print("Image saved to:", temp_path)
    image = cv2.imread(temp_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    results = detection_model(temp_path)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)

    classified_classes = []
    carbon_total = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cropped = image[y1:y2, x1:x2]
        cropped_path = f"temp_crop_{i}.jpg"
        cv2.imwrite(cropped_path, cropped)

        classification_result = classification_model(cropped_path)[0]
        class_idx = classification_result.probs.top1
        class_name = classification_result.names[class_idx]

        carbon_value = carbon_map.get(class_name.lower(), 20)
        carbon_total += carbon_value
        classified_classes.append(class_name)

        st.write(f"🌲 Tree {i+1}: **{class_name}** - {carbon_value} kg CO₂/year")

    # Pie chart
    class_counts = defaultdict(int)
    for cls in classified_classes:
        class_counts[cls.lower()] += 1

    fig, ax = plt.subplots()
    ax.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    st.pyplot(fig)

    st.success(f"🌍 Total Estimated CO₂ Sequestration: **{carbon_total:.2f} kg/year**")
