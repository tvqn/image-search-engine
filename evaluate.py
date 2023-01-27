from search import SearchEngine
from PIL import Image
import os
import glob

def Calculate_mAP(results, groundTruths):
    assert len(results) == len(groundTruths)
    mAP = 0
    for (idx, result) in enumerate(results):
        mAP = AveragePrecision(result= result, groundTruth= groundTruths[idx])
    return mAP / len(result)
def AveragePrecision(result, groundTruth):
    countTruth = 0
    AvgPrec = 0
    for (idx, item) in enumerate(result):
        if item in groundTruth:
            countTruth += 1
            AvgPrec += countTruth / (idx + 1)
    return AvgPrec / countTruth
def GetQueryImage(groundTruth, datasetPath):
    images = []
    names = []
    for query in glob.iglob(os.path.join(groundTruth, '*_query.txt')):
        query_name, x, y, w, h = open(query).read().strip().split(' ')
        names.append(query_name)
        query_name = query_name.replace('oxc1_', '') + '.jpg'
        x, y, w, h = map(float, (x, y, w, h))
        
        image = Image.open(os.path.join(datasetPath, query_name))
        images.append(image.crop((x, y, w, h)))
    return images, names
if __name__ == "__main__":
    config = {
        "groundTruthPath": os.path.join(os.getcwd(), "gt_files"),
        "datasetPath": os.path.join(os.getcwd(), "data")
    }
    images, names = GetQueryImage(
        groundTruth=config["groundTruthPath"],
        datasetPath=config["datasetPath"]
    )
    images[2].show()