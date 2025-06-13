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
st.set_page_config(page_title="Tree Carbon Estimator", layout="centered", page_icon="🌳")
st.markdown("""
    <style>
    .main {
        background-color: #e8f5e9;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌳 Tree Detection, Classification & Carbon Estimation")
st.markdown("Upload an image containing **trees**, and the app will:")
st.markdown("- 🔍 Detect each tree\n- 🏷️ Classify the species\n- 🌱 Estimate yearly CO₂ sequestration")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    image = cv2.imread(temp_path)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔎 Detecting trees..."):
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

            simple_name = name_map.get(class_name.lower(), "others")
            carbon_value = carbon_map.get(simple_name, 20)
            carbon_total += carbon_value
            classified_classes.append(simple_name)

            st.info(f"🌲 **Tree {i+1}**: `{class_name}` → {carbon_value} kg CO₂/year")

    # Pie chart
    class_counts = defaultdict(int)
    for cls in classified_classes:
        class_counts[cls] += 1

    st.markdown("---")
    st.subheader("📊 Tree Species Distribution")
    fig, ax = plt.subplots()
    ax.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown("---")
    st.success(f"🌍 **Total Estimated CO₂ Sequestration**: {carbon_total:.2f} kg/year")

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit and YOLOv8")

