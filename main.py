from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from search import SearchEngine
from PIL import Image
import io
import base64

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

@app.get("/")
async def root():
    return {"message": "search-engine-v0"}

@app.post("/search")
async def search_image(image: bytes = File(), algorithm = None):
    searchEngine = SearchEngine(indexDictPath="./sample.json")
    return {
        "code": "200",
        "message": "Success",
        "data": searchEngine.Query(image= Image.open(io.BytesIO(image)))
    }

@app.post("/search_base64")
async def search_image(image: str = Form(), algorithm = None):
    searchEngine = SearchEngine(indexDictPath="./sample.json")
    image_as_bytes = str.encode(image)
    img_recovered = base64.b64decode(image_as_bytes)
    return {
        "code": "200",
        "message": "Success",
        "data": searchEngine.Query(image= Image.open(io.BytesIO(img_recovered)))
    }