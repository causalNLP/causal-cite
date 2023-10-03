#import pandas as pd
import os
import json
import ijson
import csv

outputCSVPath = "/path/to/csv/"
datasetPath = "/path/to/dataset/"

file = open(outputCSVPath,'w',encoding="utf-8",newline='')
with file:
    header = ['id','abstract']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()

    for file in os.listdir(datasetPath):
        if file.startswith("abstract") and file.endswith(".jsonl"):
            with open(datasetPath + file,"rb") as f:
                for i,line in enumerate(f):
                    element = json.loads(line)
                    if element["corpusid"] != None:
                        d = {"id":element["corpusid"], "abstract":element["abstract"]}
                        writer.writerow(d)
                    else:
                        continue