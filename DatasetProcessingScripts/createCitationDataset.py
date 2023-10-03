import pandas as pd
import os
import json
import ijson
import csv

citationData = pd.DataFrame()

outputCSVPath = "/path/to/csv/"
datasetPath = "/path/to/dataset/"

file = open(outputCSVPath,'w',encoding="utf-8",newline='')

with file:
    header = ['from', 'to', 'influential']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()

    for file in os.listdir(datasetPath):
        if file.startswith("citation") and file.endswith(".jsonl"):
            with open(datasetPath+file,"rb") as f:
                for i,line in enumerate(f):
                    element = json.loads(line)
                    if element["citingcorpusid"] != None and element["citedcorpusid"] != None:
                        d = {"from":element["citingcorpusid"], "to":element["citedcorpusid"],"influential":element["isinfluential"]}
                        writer.writerow(d)
                    else:
                        continue
