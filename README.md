# *CausalCite*: A Causal Formulation of Paper Citations
This repo contains the code for the paper:

 [**CausalCite: A Causal Formulation of Paper Citations**](http://arxiv.org/abs/2311.02790) (2023)

*Ishan Kumar\*, Zhijing Jin\*, Ehsan Mokhtarian, Siyuan Guo, Yuen Chen, Negar Kiyavash, Mrinmaya Sachan, Bernhard Schoelkopf* (*: Co-first author)

### Usage

#### 1. Download the Semantic Scholar Dataset
To download the Dataset you need a Semantic scholar API key which lets you make 5k requests in 5 mins. This key can be obtained by filling out the form on the Semantic Scholar Website. https://www.semanticscholar.org/product/api#api-key-form.
Once you have the key, replace it in the ```downloadDatasets.py``` file in the ```apiKey``` value. 

```python
datasetDownload = "papers"
python downloadDatasets.py

datasetDownload = "abstracts"
python downloadDatasets.py
```

The Datasets will be downloaded in the form of 30 .jsonl files in the mentioned output folder

#### 2. Preprocess the Data
The 3 datasets Abstracts, Papers, Citations can be preprocessed using the scripts in ```code/dataProc/*.py```.
This will combine the relevant data from the 30 ```jsonl``` files into 1 single ```.csv```file.

#### 3. Run the CausalCite Code
Once all the Datasets are created, you can run the causal cite code by replacing the path variables.
```python
python code/getCIIdata.py --model_path /path/to/specter2 --citationData_path /path/to/fullCitationDataset.parquet.gzip --paperData_path /path/to/fullpapersdata.parquet.gzip --bm25Path /path/to/bm25_data --output_path /path/to/output --CandidatePool_path /path/to/candidate_pool --paperAid 3303339
```

The output will be created as a folder named as PaperA's ID and it will contain all of its sampled Citation Paper Bs files. These can be used to calculate the TCI.



