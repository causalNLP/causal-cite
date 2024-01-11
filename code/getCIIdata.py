import gc
import re
import pandas as pd
import pickle
import time
import dask.dataframe as dd
import torch
from rank_bm25 import *
from transformers import AutoModel, AutoTokenizer
from operator import itemgetter
import os
from collections import deque
from scipy.sparse import csr_matrix
from transformers import AutoModel
import random
import argparse

parser = argparse.ArgumentParser(
    description="Python code to calculate the TCI for a paper A."
)

parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the model folder of the Huggingface embedding model",
)
parser.add_argument("--citationData_path", type=str, help="Path to citation data file")
parser.add_argument("--paperData_path", type=str, help="Path to paper data file")
parser.add_argument("--bm25Path", type=str, help="Path to BM25 pickle")
parser.add_argument("--output_path", type=str, help="Path for output")
parser.add_argument(
    "--CandidatePool_path", type=str, help="Path to candidate pool file"
)
parser.add_argument("--paperAid", type=int, help="PaperA id to Run on")
parser.add_argument(
    "--citationDataPartition",
    type=int,
    default=70,
    help="Num Partitions for the Citation data",
)
parser.add_argument(
    "--embeddingBatchSize",
    type=int,
    default=32,
    help="Batch Size for the embedding model",
)
parser.add_argument(
    "--debugMode", type=bool, default=False, help="Debug mode for print statements"
)
parser.add_argument("--k", type=int, default=40, help="k For Averages")

args = parser.parse_args()

model_path = args.model_path
citationData_path = args.citationData_path
paperData_path = args.paperData_path
bm25Path = args.bm25Path
output_path = args.output_path
CandidatePool_path = args.CandidatePool_path
citationPartitions = args.citationDataPartition

df = dd.read_parquet(citationData_path)
df = df.repartition(npartitions=citationPartitions)

batchSize = args.embeddingBatchSize

cosineSim = torch.nn.CosineSimilarity(dim=1)
model = AutoModel.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.model_max_length = 512

df_common = pd.read_parquet(paperData_path)

DEBUG = args.debugMode
k = args.k
yearPrev = -1

start = time.time()


def calculate_max_depth(tuple_list):
    max_depth = float("-inf")  # Initialize max_depth with negative infinity
    for _, depth in tuple_list:
        if depth > max_depth:
            max_depth = depth
    return max_depth


def bfs_edge_list(edge_list, start_node):
    # Create an empty set to track visited nodes
    visited = set()
    # Create a deque for BFS traversal
    queue = deque([(start_node, 0)])  # (node, level)
    while queue:
        node, level = queue.popleft()
        # Skip already visited nodes
        if node in visited:
            continue
        # Mark current node as visited
        visited.add(node)
        # Yield the current node and its level
        # print(level)
        yield node, level
        # Get the outgoing edges from the current node
        outgoing_edges = edge_list[edge_list["to"] == node]
        # Enqueue the neighboring nodes for the next level
        for _, row in outgoing_edges.iterrows():
            neighbor = row["from"]
            queue.append((neighbor, level + 1))


def getDescendants(df, start_node):
    start = time.time()
    result = df.map_partitions(bfs_edge_list, start_node).compute(
        scheduler="threads", num_workers=30
    )
    res_list = []
    for generator in result:
        for val in generator:
            res_list.append(val)
    del result
    res_list = sorted(res_list, key=itemgetter(1))
    print("Done in :", time.time() - start)
    return res_list


def childrenFromDescendants(descendants):
    ids_with_distance_one = []
    for id, distance in descendants:
        if distance == 1:
            ids_with_distance_one.append(int(id))
    return ids_with_distance_one


def listFromDescendants(descendants):
    ids_with_distance = set()
    for id, distance in descendants:
        ids_with_distance.add(int(id))
    return list(ids_with_distance)


def countDescendants(descendants):
    unique_values = set(descendants)
    return len(unique_values)


def find_representative_numbers(numbers_with_ids, k, paperA):
    # Step 1: Sort the list based on numbers in ascending order
    paperBs_done = []
    sorted_numbers_with_ids = sorted(numbers_with_ids, key=lambda x: x[0], reverse=True)

    # Step 2: Calculate the indices of the 10 deciles
    n = max(1, len(sorted_numbers_with_ids) // k)

    papers = []
    for start_id in range(0, len(sorted_numbers_with_ids), n):
        paper_set = sorted_numbers_with_ids[start_id : start_id + n]
        this_paper = random.choice(paper_set)
        papers.append(this_paper)
    return papers


def getBM25TopK(query, subsetToSearch, k):
    tokenized_query = query.split(" ")
    docs = bm25.get_batch_scores(tokenized_query, subsetToSearch)
    top_n = np.argsort(docs)[::-1][:k]
    return top_n

def remove_mediator(text):
    # Remove all numbers
    text = re.sub(r'\b\d+(\.\d+)?%?\b', '', text)
    text = text.replace("%", "")
    
    # Specify terms to be removed
    terms_to_remove = ['state-of-the-art', 'outperforms', 'performance gains', 'large margin', 'best performance', 'enhanced']
    
    # Remove specified terms
    for term in terms_to_remove:
        text = re.sub(fr'\b{re.escape(term)}\b', '', text, flags=re.IGNORECASE)
    
    return text

paperAs = [args.paperAid]

for paperNum, paperSet in enumerate(paperAs):
    paperA = int(paperSet)
    outputRow = {
        "Role": [],
        "Abstract": [],
        "Citations": [],
        "InfCit": [],
        "PaperId": [],
        "Sim": [],
    }
    paperA = int(paperSet)
    gc.collect()
    if os.path.exists(output_path + str(paperA)):
        continue
    descendants = getDescendants(df, paperA)
    childrenOfA = childrenFromDescendants(descendants)
    lisDescendants = listFromDescendants(descendants)

    numChildrenOfA = len(childrenOfA)

    if numChildrenOfA == 0:
        print("skipping since No Children")
        continue

    dataA = df_common[df_common["id"] == paperA].head(1)

    try:
        outputRow["Role"].append("PaperA")
        outputRow["Abstract"].append(dataA["title"].item())
        outputRow["Citations"].append(dataA["citationcount"].item())
        outputRow["InfCit"].append(dataA["influentialcitationcount"].item())
        outputRow["PaperId"].append(paperA)
        outputRow["Sim"].append(dataA["year"].item())
        if DEBUG:
            print(outputRow)
    except:
        continue

    random.shuffle(childrenOfA)
    countPaperB = 0
    childSet = []

    for _paperB in childrenOfA:
        try:
            citCount = (
                df_common[df_common["id"] == _paperB].head(1)["citationcount"].item()
            )
            yearB = df_common[df_common["id"] == _paperB].head(1)["year"].item()
            childSet.append((citCount, _paperB))
        except:
            citCount = -2

    childSetFiltered = find_representative_numbers(childSet, 40, paperA)

    for _paperB in childSetFiltered:
        if _paperB == -1:
            continue
        outputRow = {
            "Role": [],
            "Abstract": [],
            "Citations": [],
            "InfCit": [],
            "PaperId": [],
            "Sim": [],
        }
        outputRow["Role"].append("PaperA")
        outputRow["Abstract"].append(dataA["title"].item())
        outputRow["Citations"].append(dataA["citationcount"].item())
        outputRow["InfCit"].append(dataA["influentialcitationcount"].item())
        outputRow["PaperId"].append(paperA)
        outputRow["Sim"].append(dataA["year"].item())

        paperB = int(_paperB[1])

        if len(df_common[df_common["id"] == paperB]) == 0:
            print("  Child not found in Dataset  ")
            continue

        paperByear = df_common[df_common["id"] == paperB].head(1)["year"].item()

        print(paperByear)

        try:
            CandidatePool = pd.read_csv(f"{CandidatePool_path}/{paperByear}.csv")

        except:
            continue

        if paperB in CandidatePool["id"].values:
            paperB_Abstract = (
                CandidatePool[CandidatePool["id"] == paperB].head(1)["abstract"].item()
            )
            paperB_Abstract = CandidatePool[CandidatePool["id"] == paperB].head(1)["title"].item() + paperB_Abstract

        elif str(paperB) in CandidatePool["id"].values:
            paperB_Abstract = (
                CandidatePool[CandidatePool["id"] == str(paperB)]
                .head(1)["abstract"]
                .item()
            )
            paperB_Abstract = CandidatePool[CandidatePool["id"] == str(paperB)].head(1)["title"].item() + paperB_Abstract

        else:
            print("Abstract Not Found, relying on Title")
            paperB_Abstract = (
                df_common[df_common["id"] == paperB].head(1)["title"].item()
            )

        dataB = df_common[df_common["id"] == paperB].head(1)

        try:
            outputRow["Role"].append("PaperB")
            outputRow["Abstract"].append(dataB["title"].item())
            outputRow["Citations"].append(dataB["citationcount"].item())
            outputRow["InfCit"].append(dataB["influentialcitationcount"].item())
            outputRow["PaperId"].append(paperB)
            outputRow["Sim"].append(dataB["year"].item())
            if DEBUG:
                print(outputRow)
        except:
            print("Paper B issue, continuing")
            continue

        start1 = time.time()
        file_path = bm25Path
        paperB_Abstract = remove_mediator(paperB_Abstract)

        if paperByear != yearPrev:
            yearPrev = paperByear
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    bm25 = pickle.load(file)
                    # bm25_sparse_matrix = csr_matrix(bm25.matrix)

            else:
                lis = []
                print("BM25 Doesnt Exist Creating")
                listOfAbstracts = CandidatePool["abstract"].tolist()
                listOfAbstracts = [remove_mediator(abstract) for abstract in listOfAbstracts]
                print("Abstracts for this Candidate Pool", len(listOfAbstracts))

                for doc in listOfAbstracts:
                    try:
                        lis.append(doc.split())
                    except:
                        lis.append([])
                del listOfAbstracts
                bm25 = BM25Okapi(lis)
                # bm25_sparse_matrix = csr_matrix(bm25.matrix)
                del lis
        else:
            print("Reusing BM25 Old")
        indexCandidatePool = pd.Index(range(len(CandidatePool)))
        # try:
        top100set_indexes_ofCP = getBM25TopK(paperB_Abstract, indexCandidatePool, 100)
        print("   T:  BM25 Load + Fetch, New Approach: ", time.time() - start1)
        encoded_PaperB = model(
            tokenizer.encode(
                paperB_Abstract, return_tensors="pt", truncation=True
            ).cuda()
        ).pooler_output

        finalCandidatePool = []
        count_goodPapers = 0
        countCandidatePoolDescendantsA = 0
        list_potentialPaper_id = []
        list_citationCount = []
        list_abstract = []
        list_infCit = []

        for paper in top100set_indexes_ofCP:
            potentialPaper = int(CandidatePool.iloc[int(paper)]["id"])
            paper = int(paper)
            paperB = int(paperB)

            if potentialPaper in lisDescendants:
                countCandidatePoolDescendantsA += 1
                print("descendent of paper A")
                continue

            if potentialPaper == paperB:
                print("Same as B")
                continue

            abstract = remove_mediator(CandidatePool.iloc[paper]["abstract"])
            title = CandidatePool.iloc[paper]["title"]
            paperCand = df_common[df_common["id"] == potentialPaper].head(1)
            potentialPaper_id = paperCand["id"].item()
            citationCount = paperCand["citationcount"].item()
            infCit = paperCand["influentialcitationcount"].item()

            list_potentialPaper_id.append(potentialPaper_id)
            list_citationCount.append(citationCount)
            list_abstract.append(title + abstract)
            list_infCit.append(infCit)

            if abstract == paperB_Abstract:
                print("Candidate Selected is same as PaperB skipping")
                continue

        full_output = []

        for batch in range(0, len(list_abstract), batchSize):
            encoded_input = tokenizer(
                list_abstract[batch : batch + batchSize],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded_input["token_type_ids"] = encoded_input["token_type_ids"].cuda()
            encoded_input["input_ids"] = encoded_input["input_ids"].cuda()
            encoded_input["attention_mask"] = encoded_input["attention_mask"].cuda()
            candidate_paper = model(**encoded_input).pooler_output
            output1 = cosineSim(encoded_PaperB, candidate_paper).tolist()
            full_output.extend(output1)

        output = full_output
        count_goodPapers = 0

        for i, _output in enumerate(output):
            if _output > 0.93:
                count_goodPapers += 1
                lis_temp = [
                    _output,
                    list_potentialPaper_id[i],
                    list_citationCount[i],
                    list_abstract[i],
                    list_infCit[i],
                ]

                if lis_temp not in finalCandidatePool:
                    finalCandidatePool.append(lis_temp)

        finalCandidatePool_sorted = sorted(
            finalCandidatePool, key=itemgetter(0), reverse=True
        )
        finalCandidatePool_sorted = finalCandidatePool_sorted[:10]

        for i, finalPaper in enumerate(finalCandidatePool_sorted):
            outputRow["Role"].append("Paper " + str(i))
            outputRow["Abstract"].append(finalPaper[3])
            outputRow["Citations"].append(finalPaper[2])
            outputRow["InfCit"].append(finalPaper[4])
            outputRow["PaperId"].append(finalPaper[1])
            outputRow["Sim"].append(finalPaper[0])

            if DEBUG:
                print(outputRow)

        if not os.path.exists(output_path + str(paperA)):
            os.mkdir(output_path + str(paperA))

        dat = pd.DataFrame(
            {
                "Role": pd.Series(outputRow["Role"]),
                "Abstract": pd.Series(outputRow["Abstract"]),
                "Citations": pd.Series(outputRow["Citations"]),
                "InfCit": pd.Series(outputRow["InfCit"]),
                "PaperId": pd.Series(outputRow["PaperId"]),
                "Sim": pd.Series(outputRow["Sim"]),
            }
        )

        dat.to_csv(
            output_path
            + str(paperA)
            + "/output_"
            + str(paperB)
            + "_"
            + str(time.time())
            + "_.csv",
            index=False,
        )
        countPaperB += 1

        del finalCandidatePool_sorted
