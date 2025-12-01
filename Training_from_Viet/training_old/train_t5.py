import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # optimizer PyTorch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
from rouge_score import rouge_scorer
from tqdm import tqdm

# Các import khác nếu cần
from dataset import SummarizationDataset
from utils import save_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_rouge(reference, predicted):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, predicted)

def train(
    train_file="/Users/trananhviet/Documents/HUMG Uni/Data Mining/SummarizeText/clean_vnexpress_dataset.json",
    val_file="/Users/trananhviet/Documents/HUMG Uni/Data Mining/SummarizeText/training/data/val.json",
    model_name="VietAI/vit5-base",
    output_dir="checkpoints",
    epochs=5,
    batch_size=2,
    lr=3e-5
):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    train_ds = SummarizationDataset(train_file, tokenizer)
    val_ds = SummarizationDataset(val_file, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in loop:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"=== Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f} ===")

        evaluate(model, tokenizer, val_loader)

        # Lưu checkpoint mỗi epoch
        save_checkpoint(model, tokenizer, f"{output_dir}/epoch_{epoch+1}")

    return model, tokenizer


def evaluate(model, tokenizer, dataloader):
    model.eval()
    rouges = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            output_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=150
            )

            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            label = tokenizer.decode(
                [x for x in batch["labels"][0].tolist() if x != -100]
            )

            rouge = compute_rouge(label, pred)
            rouges.append(rouge["rougeL"].fmeasure)

    avg_rouge = sum(rouges) / len(rouges)
    print(f"Validation ROUGE-L F1: {avg_rouge:.4f}")


if __name__ == "__main__":
    train()
