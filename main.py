import os
import cv2
import numpy as np
import mysql.connector
import faiss
from insightface.app import FaceAnalysis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face analysis
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# MySQL DB config
def connect_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='sagar',
        database='image_upload_db',
        auth_plugin="mysql_native_password"
    )

# Extract face embedding
def extract_embedding(image: np.ndarray):
    faces = face_app.get(image)
    if not faces:
        raise Exception("No face detected in the image.")
    embedding = faces[0].embedding.astype(np.float32)
    return embedding / np.linalg.norm(embedding)

# Fetch images and metadata from DB 
def fetch_data_from_db(limit=1000):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT 
            CONCAT(tai.FIRST_NAME, ' ', tai.LAST_NAME) AS accused_name, 
            tai.RELATIVE_NAME,
            CONCAT(SUBSTR(tfr.FIR_REG_NUM, -4), '/', tfr.REG_YEAR) AS fir_reg_num,
            tai.ACCUSED_SRNO,
            taf.UPLOADED_FILE,
            taf.FILE_NAME
        FROM t_accused_info tai
        INNER JOIN t_fir_registration tfr ON tai.fir_reg_num = tfr.fir_reg_num 
        LEFT JOIN t_accused_files taf ON taf.ACCUSED_SRNO = tai.ACCUSED_SRNO
        WHERE taf.UPLOADED_FILE IS NOT NULL
        LIMIT {limit}
    """)

    records = cursor.fetchall()
    cursor.close()
    conn.close()

    valid_data = []
    for accused_name, relative_name, fir_reg_num, accused_srno, uploaded_file, file_name in records:
        try:
            if isinstance(uploaded_file, str) and uploaded_file.startswith("data:image"):
                uploaded_file = base64.b64decode(uploaded_file.split(",")[1])
            elif isinstance(uploaded_file, (bytes, bytearray)):
                uploaded_file = bytes(uploaded_file)
            else:
                raise ValueError("Unsupported image format")

            valid_data.append({
                "accused_name": accused_name,
                "relative_name": relative_name,
                "fir_reg_num": fir_reg_num,
                "accused_srno": accused_srno,
                "file_name": file_name,
                "image_bytes": uploaded_file
            })
        except Exception as e:
            logger.warning(f"Skipping invalid image {file_name}: {e}")

    return valid_data

# Input model
class ImageData(BaseModel):
    image_base64: str

# Face search endpoint
@app.post("/search")
async def search_image_base64(data: ImageData):
    try:
        base64_str = data.image_base64.split(",")[-1]
        img_data = base64.b64decode(base64_str)
        query_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if query_img is None:
            return {"error": "Invalid image format."}

        query_embedding = extract_embedding(query_img)
        dim = len(query_embedding)

        db_data = fetch_data_from_db()
        if not db_data:
            return {"error": "No images in the database."}

        embeddings, metadata = [], []
        for item in db_data:
            db_img = cv2.imdecode(np.frombuffer(item["image_bytes"], np.uint8), cv2.IMREAD_COLOR)
            if db_img is None:
                continue
            try:
                emb = extract_embedding(db_img)
                embeddings.append(emb)
                metadata.append(item)
            except Exception as e:
                logger.warning(f"Embedding failed for {item['file_name']}: {e}")

        if not embeddings:
            return {"error": "No valid face embeddings in the database."}

        index = faiss.IndexFlatIP(dim)
        index.add(np.array(embeddings, dtype=np.float32))
        
        D, I = index.search(np.array([query_embedding]), k=10)
        threshold = 0.4

        matches = []
        for dist, idx in zip(D[0], I[0]):
            if dist >= threshold:
                item = metadata[idx]
                matches.append({
                    "accused_srno": item["accused_srno"],
                    "accused_name": item["accused_name"],
                    "relative_name": item["relative_name"],
                    "fir_reg_num": item["fir_reg_num"],
                    "file_name": item["file_name"],
                    "similarity": float(dist),
                    "image_base64": base64.b64encode(item["image_bytes"]).decode("utf-8")
                })

        if not matches:
            return {"message": "No matching faces found within threshold."}

        return {"matches": matches}

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": str(e)}

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
