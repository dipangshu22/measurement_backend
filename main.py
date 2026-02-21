from fastapi import FastAPI, UploadFile, File
import os
from vision_processor import process_image

app = FastAPI()


@app.post("/measure")
async def measure(file: UploadFile = File(...)):
    file_path = "temp.jpg"

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        height, gesture = process_image(file_path)

        if height is None:
            return {"error": "Body not detected clearly."}

        weight = 22 * ((height / 100) ** 2)

        os.remove(file_path)

        return {
            "estimated_height_cm": height,
            "estimated_weight_kg": round(weight, 2),
            "gesture_detected": gesture
        }

    except Exception as e:
        return {"error": str(e)}