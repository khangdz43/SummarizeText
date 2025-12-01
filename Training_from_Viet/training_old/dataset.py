import json
from torch.utils.data import Dataset

class SummarizationDataset(Dataset):
    def __init__(self, path, tokenizer, max_input_len=512, max_output_len=128):
        self.data = json.load(open(path))
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source = item["article"]
        target = item["summary"]

        model_input = self.tokenizer(
            source, truncation=True, padding="max_length",
            max_length=self.max_input_len, return_tensors="pt"
        )

        label_input = self.tokenizer(
            target, truncation=True, padding="max_length",
            max_length=self.max_output_len, return_tensors="pt"
        )

        model_input = {k: v.squeeze() for k, v in model_input.items()}
        label_ids = label_input["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_input["input_ids"],
            "attention_mask": model_input["attention_mask"],
            "labels": label_ids
        }
