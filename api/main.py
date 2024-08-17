import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MODEL = tf.keras.layers.TFSMLayer("../saved_models/1", call_endpoint="serving_default")
MODEL = tf.keras.models.load_model("../saved_models/x_ray.h5")
CLASS_NAMES = ["Normal", "PNEUMONIA"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = cv2.cvtColor(read_file_as_image(await file.read()),cv2.COLOR_GRAY2RGB)
    IMAGE_SIZE = 256

    img = tf.keras.preprocessing.image.smart_resize(
        image,

        size=(IMAGE_SIZE, IMAGE_SIZE),

    )
    img_batch = np.expand_dims(img, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
