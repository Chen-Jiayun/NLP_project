import os
import json

def get_title(name):
    with open(name, "r", encoding="utf-8") as file:
        data = json.load(file)
    titles = data["metadata"]["title"]
    abstract = data.get("abstract", None)
    print(titles)
    if abstract:
        print(abstract)

def process_data(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                get_title(file_path)

# export NLP_DATA_PATH="..."

try:
    data_dir_value = os.environ["NLP_DATA_PATH"]
    process_data(data_dir_value)

except KeyError:
    print("environment variable is not set.")
    exit(1)
