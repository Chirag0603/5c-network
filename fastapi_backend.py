from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# Load the trained model (make sure the model file is in the same directory)
model = load_model('unet_plus_plus_model.h5')

# Preprocess image for the model (CLAHE, normalization, resizing, etc.)
def preprocess_image(image):
    # Convert the PIL image to a NumPy array and convert to grayscale
    image = np.array(image.convert('L'))

    # Apply CLAHE for preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Normalize the image
    image = image / 255.0

    # Resize image to the input size expected by the model (e.g., 256x256)
    image = cv2.resize(image, (256, 256))

    # Reshape to add batch and channel dimensions (1, 256, 256, 1)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension

    return image

# FastAPI endpoint for predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict segmentation mask
    prediction = model.predict(preprocessed_image)

    # Post-process prediction (reshape, thresholding)
    prediction = (prediction > 0.5).astype(np.uint8)  # Binary mask thresholding
    prediction = np.squeeze(prediction)  # Remove single dimensions

    # Convert prediction to an image format (e.g., PNG)
    output_image = Image.fromarray((prediction * 255).astype(np.uint8))  # Scale back to [0, 255]

    # Save the output as a PNG file in memory
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)

    return {
        "filename": file.filename,
        "content": buf.read()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
