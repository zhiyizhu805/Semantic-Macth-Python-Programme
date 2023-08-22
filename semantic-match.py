import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import sys
import os

# Create source and output directories if they don't exist
source_dir = 'source'
output_dir = 'output'
os.makedirs(source_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Open a process log file to record the process
log_file_path = os.path.join(output_dir, "process_log.txt")
log_file = open(log_file_path, "w")
sys.stdout = log_file

# Load and preprocess data
dataitem_file_path = os.path.join(source_dir, '#######.xlsx')   #1st schema excel file name
dataschema_file_path = os.path.join(source_dir, '######.xlsx')  #2nd schema excel file name
dataitem_data = pd.read_excel(dataitem_file_path)
dataschema_data = pd.read_excel(dataschema_file_path)


# Load pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define function to compute similarity
def compute_semantic_similarity(sentences1, sentences2):
    inputs1 = bert_tokenizer(sentences1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = bert_tokenizer(sentences2, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings1 = bert_model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = bert_model(**inputs2).last_hidden_state.mean(dim=1)
    similarity_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    return similarity_scores.numpy()

# Compute semantic similarity between DataItem and DataSchema
has_schema_relations = []
threshold = 0.70
# the threshold can be adjusted,I set it the 0.70 since the lowest similarity_scores being calculated for
# the pastureCover instances are just greater than 0.70.

# Using tqdm to display a progress bar
for individual_name, comment in tqdm(dataitem_data[['individual_name', 'comment']].itertuples(index=False), total=len(dataitem_data)):
    print(f"Processing DataItem individual: {individual_name}")
    sentences1 = f"{individual_name} {comment}"
    print('Desc:',sentences1)
    related_instances = []
    for idx, row in dataschema_data.iterrows():
        instance_name = row['Instance']
        high_level_summary = row['High-Level Summary']
        subclass_name = row['3Sub-Class']
        sentences2 = [f"{subclass_name} {instance_name} {high_level_summary}"]
        similarity_scores = compute_semantic_similarity(sentences1, sentences2)
        print(f"Comparing: {instance_name}({similarity_scores})")
        if similarity_scores[0] > threshold:
            related_instances.append(dataschema_data['Instance'].iloc[idx])
    print(f"Related schema instances: {related_instances}")
    has_schema_relations.append(','.join(related_instances))

# Update DataItem table and save
dataitem_data['hasSchema'] = has_schema_relations
updated_dataitem_file_path = os.path.join(output_dir, 'Updated_DataItem-All-Individuals.xlsx')
dataitem_data.to_excel(updated_dataitem_file_path, index=False)

# Close the process log file at the end
log_file.close()
sys.stdout = sys.__stdout__