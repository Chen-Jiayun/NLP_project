# NLP-Project for 2023 Data Science Summer School Imperial College London

Contributors: Chen Jiayun, Liu Ziyang, Wang Yiyang

Guide for project is in **word_representations_biomedical.ipynb**, **cord19WordVectors.ipynb**

## Data Preparation

### Get data

we get the data from :

```shell
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-07-26/document_parses.tar.gz
```

decompress the file after you get it.

### Set environment variable

It's **vital** to set the environment variable **NLP_DATA_PATH** before make step-1, simply set at the root directory of the data file above is acceptable, step-1.py will scan all the subdirectories and convert all the files ended up with .json.

```bash
export NLP_DATA_PATH="/path/to/document_parse"
```

## Explanation for each step

### Step 1

We need to extract all JSON files from a folder to parse raw text. We use the 'os.walk' API to scan all files (including subdirectories) in a specific directory and extract the required content from them. Due to device limitations, we only extracted the 'title' and 'abstract' parts.

### Step 2

We utilized three methods for tokenization, which include using Python's built-in split function, the NLTK library, and the ByteLevelBPETokenizer library. The extracted results were output to the "result" folder (located outside of this GitHub repository).

Check the detail code in Makefile or type **make step-2-{method}**

### Step 3
We trained two kind of models

#### N-gram model

The definition of the model is set in **ngram_model.py**, some magic number such as dimension of the vector and the particular N for the N-gram。

The code here is primarily derived from this document, 

```text
https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
```

and we made slight modifications to make it compatible with our tokens.


#### skip-gram model
The definition of the model is set in **skip_gram_model.py**.

Actually, we just made change on the guide above and changed it in to another type of training.

#### Output

There are two kind of output: word embedding and model save.

Check the **result** directory. The file ended with **.json** contains the token and its vector.

Check the **model** directory. The model is saved for further training.

Both of the directories is not included in the git repository, but it will automatically created when you run **make step-3-ngram** or **make step-3-skipgram**

### Step 4

#### tSNE
we use t-SNE and plot draw all the vectors on a 2D graph

#### bio-t-SNE
We only plotted a subset of points related to biomedical data to examine the overall distribution trend of the vectors.

#### co-occurrence
We determine which words often co-occur by calculating the correlation coefficients.

#### similar
We determine which words have similarity by calculating the dot product of their vectors.


## Summary
Even with a relatively small dataset and limited computational resources, we can still observe that the overall performance of the model has improved significantly. Although the fine-grained classification is not yet accurate, we can trust that its performance will be better with increased data and computational power.