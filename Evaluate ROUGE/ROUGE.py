from rouge_score import rouge_scorer

# Đọc file
with open("/Users/trananhviet/Documents/HUMG Uni/Data Mining/BTL_SUMMARIZE_TEXT/Evaluate ROUGE/candidate.txt", "r", encoding="utf-8") as f:
    candidate = f.read()

with open("/Users/trananhviet/Documents/HUMG Uni/Data Mining/BTL_SUMMARIZE_TEXT/Evaluate ROUGE/reference.txt", "r", encoding="utf-8") as f:
    reference = f.read()

# Khởi tạo scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Tính ROUGE
scores = scorer.score(reference, candidate)

# In kết quả
print("ROUGE-1:", scores['rouge1'])
print("ROUGE-2:", scores['rouge2'])
print("ROUGE-L:", scores['rougeL'])
