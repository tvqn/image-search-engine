from fastapi import FastAPI, File, UploadFile
from search import SearchEngine
from PIL import Image
import io

app = FastAPI()

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
