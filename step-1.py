import os
import json

def get_raw_text(name):
    with open(name, "r", encoding="utf-8") as file:
        data = json.load(file)
    titles = data["metadata"]["title"]
    abstract = data.get("abstract", None)
    abstract_text = []
    print(titles)

    # due to the limited device, we parse the titles and abstracts
    # you may select more parts of the json file.
    if abstract:
        for item in abstract:
            abstract_text.append(item.get("text",None))
        for item_text in abstract_text:
            print(item_text)

def process_data(path):
    # check file recursively through os.walk
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                get_raw_text(file_path)

# export NLP_DATA_PATH="..."

try:
    data_dir_value = os.environ["NLP_DATA_PATH"]
    process_data(data_dir_value)

except KeyError:
    print("environment variable is not set.")
    exit(1)