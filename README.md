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


