# 🌳 Tree Detection, Classification & Carbon Sequestration Estimator

This Streamlit app allows users to upload aerial or landscape images containing trees. The application performs:

1. **Tree Detection** using a YOLOv8 object detection model
2. **Tree Species Classification** using a fine-tuned YOLOv8 classification model
3. **Carbon Sequestration Estimation** based on the detected tree types

## 🚀 Try the App
**Live demo**: [Click to Launch App](https://tree-app-aca5n5qupcbngzstflzi7y.streamlit.app/)

---

## 🧠 How It Works

- **Detection Model (`detection.pt`)**  
  Locates individual trees within the uploaded image.

- **Classification Model (`classification.pt`)**  
  Identifies the species of each tree (e.g., Neem, Banyan, Mango, etc.).

- **Carbon Estimation**  
  Each tree type is mapped to an approximate annual carbon sequestration value (in kg CO₂/year) based on literature and studies. The total is computed and shown for the entire image.

## 🧾 Supported Tree Types and Their CO₂ Values

| Tree Species     | Annual CO₂ Absorption (kg) |
|------------------|-----------------------------|
| Neem             | 26                          |
| Banyan           | 40                          |
| Coconut          | 35                          |
| Amla             | 18.5                        |
| Mango            | 30                          |
| Pine             | 22                          |
| Others (default) | 20                          |

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- PIL
- Matplotlib
- YOLOv8 (Ultralytics)
- PyTorch
- ## 👨‍💻 Author

**Mayank Kumar Sharma**  
> Developed as part of an AI-driven environmental project for real-world tree monitoring and carbon impact assessment.
