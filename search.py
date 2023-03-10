
import json
import imagehash
from PIL import Image
import io
import shutil
import os
import h5py
from tqdm import tqdm
import cv2
import numpy as np

from fisher_vector import FisherVectorExtraction, QuerySift

class FeatureExtraction:
    def __init__(self, config: dict = None) -> None:
        self.DataPath = os.path.join(os.getcwd(), "data")
        self.CollectorPath = os.path.join(os.getcwd(), )
        self.TypeFeature = None
        self.GetConfig(config)
        pass

    def GetConfig(self, config: dict) -> None:
        if config == None:
            return
        if "dataPath" in config.keys() and config["dataPath"] != None:
            self.DataPath = config["dataPath"]
        if "collectorPath" in config.keys() and config["collectorPath"] != None:
            self.CollectorPath = config["collectorPath"]
        if "typeFeature" in config.keys() and config["typeFeature"] != None:
            self.TypeFeature = config["typeFeature"]

    def FeatureExtraction(self, image, typeFeature=None):
        return imagehash.average_hash(image=image)

    def SiftFeatureExtraction(self, pil_image, imageID, savePath):
        sift = cv2.xfeatures2d.SIFT_create()
        image = self.ToImageOpenCV(image=pil_image, grayScale=True)
        height, width = image.shape[:2]
        img_resize = cv2.resize(image, (int(0.5*width), int(0.5*height)))
        kp, des = sift.detectAndCompute(img_resize, None)
        
        with open(os.path.join(savePath, imageID + '.opencv.sift'), 'w') as fileSiftFeature:
            if des is None:
                fileSiftFeature.write(str(128) + '\n')
                fileSiftFeature.write(str(0) + '\n')
            elif len(des) > 0:
                fileSiftFeature.write(str(128) + '\n')
                fileSiftFeature.write(str(len(kp)) + '\n')
                for j in range(len(des)):
                    locs_str = '0 0 0 0 0 '
                    descs_str = " ".join([str(int(value)) for value in des[j]])
                    all_strs = locs_str + descs_str
                    fileSiftFeature.write(all_strs + '\n')
        # img = cv2.drawKeypoints(img_resize, kp, img_resize)
        # return img

    def ToImageOpenCV(self, image, grayScale=False):
        npImage = np.array(image)
        if grayScale == True:
            return cv2.cvtColor(npImage, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(npImage, cv2.COLOR_RGB2BGR)

    def BuildFeatureCollector(self):
        data = {}
        for imgName in tqdm(os.listdir(self.DataPath), desc="Build feature collector"):
            imgPath = os.path.join(self.DataPath, imgName)
            image = Image.open(imgPath)
            data[imgName] = {
                "feature": str(self.FeatureExtraction(image, self.TypeFeature)),
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

    def Query(self, image: Image, topK: int = 10, typeFeature = None, verbose: bool = False):
        getTop = []
        if typeFeature == "sift":
            configQuery = {
                "topK": 10,
                "gmmSavePath": os.path.join(os.getcwd(), "tmp", "gmm.pkl"),
                "featureSavePath": os.path.join(os.getcwd(), "tmp", 'fisher.h5')
            }
            siftQueryRslt = QuerySift(image= image, config= configQuery)
            for item in siftQueryRslt:
                imageID = item[1].decode("utf-8") + '.jpg'
                getTop.append([
                    item[0],
                    imageID,
                    self.IndexDict[imageID]['size']
                ])
        else:
            hash = imagehash.average_hash(image)
            # Metric algorithm
            calculateSimilarRslt = []
            for i in tqdm(self.IndexDict.keys(), desc="Query", disable=not verbose):
                similarRslt = self.CalculateSimilar(
                    i, hash, imagehash.hex_to_hash(self.IndexDict[i]['feature']))
                calculateSimilarRslt.append(
                    [similarRslt[0], similarRslt[1], self.IndexDict[i]['size']])
                # #Hammington distance
                # score.append(hash - imagehash.hex_to_hash(self.IndexDict[i]['feature']))
                # name.append(i)
            # Ranking
            # for img in sorted(zip(score, name))[:10]:
            #     src = os.path.join('data', img[1])
            #     dst = 'result'
            #     shutil.copy(src, dst)
            # return self.Ranking(zip(score, name), 10);
            getTop = self.Ranking(result=calculateSimilarRslt, topK=topK)

        result = []
        for item in getTop:
            result.append({
                "name": item[1],
                "score": item[0],
                "size": {
                    "width": item[2][0],
                    "height": item[2][1]
                }
            })
        return result

    def CalculateSimilar(self, name, featureQuery, featureCollector, typeFeature=None):
        # Hammington distance
        score = featureQuery - featureCollector
        return (score, name)

    def Ranking(self, result, topK, typeFeature=None):
        return sorted(result)[:topK]

if __name__ == "__main__":
    config = {
        "dataPath": os.path.join(os.getcwd(), "data"),
        "collectorPath": os.path.join(os.getcwd(), "sample1.json"),
        "typeFeature": None
    }
    featureExtraction = FeatureExtraction(config=config)

    imageID = "all_souls_000001"
    siftFeaturePath = os.path.join(os.getcwd(), "sift_feature")
    imagePath = os.path.join(os.getcwd(), "data", imageID + ".jpg")
    image = Image.open(imagePath)
    sift = featureExtraction.SiftFeatureExtraction(image, imageID = imageID, savePath = siftFeaturePath)
    # opencv_image = featureExtraction.ToImageOpenCV(image=image, grayScale=True)
    # cv2.imshow("OpenCV Image", sift)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # featureExtraction.BuildFeatureCollector()
