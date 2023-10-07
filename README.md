# CausalCite : A Causal Formulation of Paper Citations 
![Graph_CausalCite](https://github.com/causalNLP/causal-cite/assets/46553104/a1e2e1a6-2418-41bb-b472-6ef844f6a496)

### Introduction: 
Evaluating the significance of a paper is pivotal yet challenging for the scientific community. While the citation count is the most commonly used proxy for this purpose, they are widely criticized for failing to accurately reflect a paper’s true impact. In this work, we propose a causal inference method, SYNMATCH, which adapts the traditional matching framework to high-dimensional text embeddings. Specifically, we encode each paper using the text embeddings by large language models, extract similar samples by cosine similarity, and synthesize a counterfactual sample by the weighted average of similar papers according to their similarity values.

### Usage

1. Downloading the Semantic Scholar Dataset
To download the Dataset you need a Semantic scholar API key which lets you make 5k requests in 5 mins. This key can be obtained by filling out the form on the Semantic Scholar Website. https://www.semanticscholar.org/product/api#api-key-form.
Once you have the key, replace it in the ```downloadDatasets.py``` file in the ```apiKey``` value. 
```python
datasetDownload = "papers"
python downloadDatasets.py

datasetDownload = "abstracts"
python downloadDatasets.py
```

The Datasets will be downloaded in the form of 30 .jsonl files in the mentioned output folder

2. Preprocess the Data
The 3 datasets Abstracts, Papers, Citations can be preprocessed using the scripts in ```/DatasetProcessingScripts/```.
This will combine the relevant data from the 30 ```jsonl``` files into 1 single ```.csv```file.

3. Run the CausalCite Code
Once all the Datasets are created, you can run the causal cite code by replacing the path variables.
```python
python getCIIdata.py
```

The output will be created as a folder named as PaperA's ID and it will contain all of its sampled Citation Paper Bs files. These can be used to calculate the TCI.



