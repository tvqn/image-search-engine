from search import SearchEngine
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm

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
    query_names = []
    prefix_file_query_names = []
    for query in glob.iglob(os.path.join(groundTruth, '*_query.txt')):
        prefix_file_query_name = os.path.splitext(os.path.basename(query))[0].replace('_query', '')
        prefix_file_query_names.append(prefix_file_query_name)

        query_name, x, y, w, h = open(query).read().strip().split(' ')
        query_name = query_name.replace('oxc1_', '') + '.jpg'
        query_names.append(query_name)
        x, y, w, h = map(float, (x, y, w, h))
        
        image = Image.open(os.path.join(datasetPath, query_name))
        images.append(image.crop((x, y, w, h)))
    return images, query_names, prefix_file_query_names
if __name__ == "__main__":
    config = {
        "groundTruthPath": os.path.join(os.getcwd(), "gt_files"),
        "datasetPath": os.path.join(os.getcwd(), "data")
    }
    images, query_names, prefix_file_query_names = GetQueryImage(
        groundTruth=config["groundTruthPath"],
        datasetPath=config["datasetPath"]
    )
    # images[2].show()
    configSearchEngine = {
        "featurePath": os.path.join(os.getcwd(), "sample.json")
    }
    searchEngine = SearchEngine(configSearchEngine["featurePath"])

    aps = []
    rank_file = os.getcwd() + "tmp.txt"
    for idx, query in enumerate(tqdm(images, desc="Calculate mAP")):
        result = searchEngine.Query(image= images[idx], topK= 10, typeFeature="sift")

        with open(rank_file, 'w') as f:
            f.writelines([name['name'][1].split('.jpg')[0] + '\n' for name in result])
        # compute mean average precision
        gt_prefix = os.path.join(config["groundTruthPath"], prefix_file_query_names[idx])
        cmd = 'compute_ap %s %s' % (gt_prefix, rank_file)
        ap = os.popen(cmd).read()
        aps.append(float(ap.strip()))
        
    print("mAP:", np.array(aps).mean())