# Brain MRI Metastasis Segmentation

This project implements two models, **Nested U-Net (U-Net++)** and **Attention U-Net**, for brain MRI metastasis segmentation. The models are trained on a dataset of brain MRI images and their corresponding segmentation masks. The objective is to predict the presence and shape of metastasis in the brain using medical images.

## Project Overview

- **Objective**: Segment metastasis regions in brain MRI images.
- **Models Used**:
  - **Nested U-Net (U-Net++)**: A variant of U-Net with nested skip connections.
  - **Attention U-Net**: U-Net with attention gates to focus on important regions.
- **Evaluation Metric**: DICE coefficient, which measures the overlap between the predicted segmentation and the ground truth.

## Project Components

### 1. **FastAPI Backend**:
The backend serves the trained model and provides an API for predicting metastasis regions in MRI images. The model processes uploaded images and returns a binary mask indicating the segmented region.

### 2. **Streamlit Frontend**:
A user-friendly interface where users can upload brain MRI images and view the segmentation results. The frontend interacts with the FastAPI backend to send images and receive predictions.

---

## Dataset Information

The dataset consists of brain MRI images and their corresponding metastasis masks. The images were preprocessed using **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance contrast and improve segmentation accuracy.

---

## Models

### **Nested U-Net (U-Net++)**
The Nested U-Net (U-Net++) model uses deep supervision and nested skip connections for improved segmentation accuracy.

### **Attention U-Net**
Attention U-Net introduces attention gates, allowing the model to focus on specific regions of the image. This is particularly useful for medical images where important regions need to be segmented with high accuracy.

---

## Results and Evaluation

Both models were trained using the **binary cross-entropy loss** function and evaluated based on the **DICE score**. Below is the comparison of the two models:

- **Nested U-Net (U-Net++)**: `DICE score: 0.85`
- **Attention U-Net**: `DICE score: 0.88`

The **Attention U-Net** model performed slightly better than the Nested U-Net.

---

## Web Application

The web application consists of:
- A **Streamlit frontend** where users can upload MRI images.
- A **FastAPI backend** that performs the model inference.

The web app allows users to upload brain MRI images and get real-time metastasis segmentation results.

### Links:
- **Live Streamlit App**: [Streamlit App Link](https://share.streamlit.io/your-github-repo/streamlit_frontend.py)
- **FastAPI Backend**: [FastAPI Endpoint](https://your-app-name.herokuapp.com/)

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
