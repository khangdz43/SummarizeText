import torch

def save_checkpoint(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def load_json_data(path):
    import json
    return json.load(open(path))
