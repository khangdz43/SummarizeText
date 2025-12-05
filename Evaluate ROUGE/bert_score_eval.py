from bert_score import score

# Hàm đọc file .txt
def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Đọc văn bản từ file
reference_text = read_txt("/Users/trananhviet/Documents/HUMG Uni/Data Mining/BTL_SUMMARIZE_TEXT/Evaluate ROUGE/reference.txt")
summary_text = read_txt("/Users/trananhviet/Documents/HUMG Uni/Data Mining/BTL_SUMMARIZE_TEXT/Evaluate ROUGE/candidate.txt")

# Đưa vào dạng list để phù hợp thư viện BERTScore
references = [reference_text]
candidates = [summary_text]

# Tính BERTScore
P, R, F1 = score(candidates, references, lang="vi")

print("===== KẾT QUẢ BERTScore =====")
print("Precision:", round(P.mean().item(), 4))
print("Recall:   ", round(R.mean().item(), 4))
print("F1-score: ", round(F1.mean().item(), 4))
