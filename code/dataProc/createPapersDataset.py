#import pandas as pd
import os
import json
import ijson
import csv

outputCSVPath = "/path/to/csv/"
datasetPath = "/path/to/dataset/"

file = open(outputCSVPath,'w',encoding="utf-8",newline='')

with file:
    header = ['id','year','citationcount','influentialcitationcount','referencecount','title']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()

    for file in os.listdir(datasetPath):
        if file.startswith("papers") and file.endswith(".jsonl"):
            #print(file)
            with open(datasetPath + file,"rb") as f:
                for i,line in enumerate(f):
                    element = json.loads(line)
                    if element["corpusid"] != None and element["citationcount"] != None and element["year"]!=None:
                        d = {"id":element["corpusid"], "year":element["year"], "citationcount":element["citationcount"], "influentialcitationcount":element["influentialcitationcount"], "referencecount":element["referencecount"],"title":element["title"]}
                        writer.writerow(d)
                        
                    else:
                        continue