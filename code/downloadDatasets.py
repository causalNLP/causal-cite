import urllib
import requests
import time

apiKey = ""
datasetDownload = "papers"
outputPath = "/path/to/output/"

if datasetDownload == "papers":
    for i in range(31):
        papers = requests.get("http://api.semanticscholar.org/datasets/v1/release/2023-04-06/dataset/papers",headers={"x-api-key":apiKey}).json()
        print(f"Downloading {i} From {papers['files'][i]}")
        urllib.request.urlretrieve(papers['files'][i], outputPath + f"data_papers-part{i}.jsonl.gz")
        #time.sleep(600)

elif datasetDownload == "abstracts":
    for i in range(35):
        papers = requests.get("http://api.semanticscholar.org/datasets/v1/release/2023-04-06/dataset/abstracts",headers={"x-api-key":apiKey}).json()

        print(f"Downloading {i} From {papers['files'][i]}")
        urllib.request.urlretrieve(papers['files'][i],  outputPath + f"abstracts-part{i}.jsonl.gz")
        #time.sleep(600)