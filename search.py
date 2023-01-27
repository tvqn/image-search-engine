
import json
import imagehash
from PIL import Image
import io
import shutil
import os
from tqdm import tqdm

class FeatureExtraction:
    def __init__(self, config : dict = None) -> None:
        self.DataPath = os.path.join(os.getcwd(), "data")
        self.CollectorPath = os.path.join(os.getcwd(), )
        self.TypeFeature = None
        self.GetConfig(config)
        pass
    def GetConfig(self, config : dict) -> None:
        if config == None:
            return
        if "dataPath" in config.keys() and config["dataPath"] != None:
            self.DataPath = config["dataPath"]
        if "collectorPath" in config.keys() and config["collectorPath"] != None:
            self.CollectorPath = config["collectorPath"]
        if "typeFeature" in config.keys()  and config["typeFeature"] != None:
            self.TypeFeature = config["typeFeature"]
    def FeatureExtraction(self, image, typeFeature = None):
        return imagehash.average_hash(image=image)
    def BuildFeatureCollector(self):
        data = {}
        for imgName in tqdm(os.listdir(self.DataPath), desc="Build feature collector"):
            imgPath = os.path.join(self.DataPath, imgName)
            image = Image.open(imgPath)
            data[imgName] = {
                "feature" : str(self.FeatureExtraction(image, self.TypeFeature)),
                "name": imgName,
                "size": [image.width, image.height]
            }
        
        json_object = json.dumps(data, indent=4)
        with open(self.CollectorPath, "w") as outfile:
            outfile.write(json_object)
class SearchEngine:
    def __init__(self, indexDictPath):
        self.IndexDictPath = indexDictPath
        self.IndexDict = None

        self.LoadIndex(indexDictPath)
    def GetConfig():
        pass
    def LoadIndex(self, path):
        with open(path) as indexFile:
            self.IndexDict = json.load(indexFile)
    def Query(self, img):
        score = []
        name = []
        image = Image.open(io.BytesIO(img))
        hash = imagehash.average_hash(image)
        #Metric algorithm
        for i in tqdm(self.IndexDict.keys(), desc="Query"):
            #Hammington distance
            score.append(hash - imagehash.hex_to_hash(self.IndexDict[i]['feature']))
            name.append(i)
        #Ranking
        # for img in sorted(zip(score, name))[:10]:
        #     src = os.path.join('data', img[1])
        #     dst = 'result'
        #     shutil.copy(src, dst)
        return self.Ranking(zip(score, name), 10);
    def CalculateSimilar(self, name, featureQuery, featureCollector, typeFeature = None):
        score = featureQuery - featureCollector
        return (name, score)
    def Ranking(self, result, amount, typeFeature = None):
        return sorted(result)[:amount]
    
if __name__ == "__main__":
    config = {
        "dataPath": None,
        "collectorPath": os.path.join(os.getcwd(), "sample1.json"),
        "typeFeature": None
    }
    featureExtraction = FeatureExtraction(config= config)
    featureExtraction.BuildFeatureCollector()