import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from sklearn.preprocessing import normalize


def get_data(file_path: str) -> str:
   
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def get_stopwords(file_path_stopwords: str) -> set:
   
    with open(file_path_stopwords, encoding="utf-8") as f:
        words = f.read().splitlines()
    return set([w.strip() for w in words if w.strip()])


def clean_text(text: str) -> str:
    
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\n", " ")
    return " ".join(text.split())


def split_sentences(text: str) -> list:
   
    if not text:
        return []
    return nltk.sent_tokenize(text)


def load_w2v_model(path: str, binary: bool = False) -> KeyedVectors:
    try:
        return KeyedVectors.load(path)
    except Exception:
        # try opposite mode
        return KeyedVectors.load(path, binary=not binary)


def sentences_to_vectors(sentences: list, w2v: KeyedVectors, stopwords: set) -> np.ndarray:
  
    if not sentences:
        return np.empty((0, w2v.vector_size), dtype=np.float32)

    vector_size = w2v.vector_size
    X = []
    for sent in sentences:
        sent_tok = ViTokenizer.tokenize(sent)
        words = [w for w in sent_tok.split() if w and w not in stopwords]
        vecs = [w2v[w] for w in words if w in w2v.key_to_index]
        if not vecs:
            X.append(np.zeros(vector_size, dtype=np.float32))
        else:
            X.append(np.mean(vecs, axis=0))
    return np.vstack(X)


def cluster_sentence_vectors(X: np.ndarray, n_clusters: int = None, ratio: float = 0.3, random_state: int = 42, normalize_vectors: bool = True) -> KMeans:
  
    n_sent = X.shape[0]
    if n_sent == 0:
        return None
    if n_clusters is None:
        suggested = max(1, int(np.ceil(n_sent * ratio)))
        n_clusters = min(suggested, n_sent)

    vecs = X
    if normalize_vectors:
        vecs = normalize(vecs)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(vecs)
    return kmeans


def build_summary(kmeans: KMeans, sent_vecs: np.ndarray, sentences: list) -> str:

    if kmeans is None or sent_vecs.size == 0:
        return ""

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sent_vecs)

    n_clusters = kmeans.n_clusters
    avg_positions = []
    for i in range(n_clusters):
        members = np.where(kmeans.labels_ == i)[0]
        if len(members) == 0:
            avg_positions.append(float('inf'))
        else:
            avg_positions.append(float(np.mean(members)))

    cluster_order = sorted(range(n_clusters), key=lambda c: avg_positions[c])

    chosen_idxs = []
    for c in cluster_order:
        idx = int(closest[c])
        if idx not in chosen_idxs:
            chosen_idxs.append(idx)

    # Keep sentences in original order
    chosen_idxs = sorted(chosen_idxs)
    summary_sents = [sentences[i].strip() for i in chosen_idxs]
    return " ".join(summary_sents)


def summarize_text(contents: str, w2v_path: str = 'wiki.vi.kv', stopwords_path: str = 'vietnamese-stopwords.txt', num_sentences: int = 3) -> str:
    """High-level pipeline: clean -> tokenize -> compute vectors -> cluster -> pick summary.

    This is a convenience wrapper you can call directly.
    """
    stop_words = get_stopwords(stopwords_path) if stopwords_path else set()
    text = clean_text(contents)
    sentences = split_sentences(text)
    if not sentences:
        return ""

    w2v = load_w2v_model(w2v_path, binary=False)
    sent_vecs = sentences_to_vectors(sentences, w2v, stop_words)

    # fallback if too few sentences or vectors are zero
    if sent_vecs.shape[0] <= num_sentences or np.allclose(sent_vecs, 0):
        return ' '.join([s.strip() for s in sentences[:num_sentences]])

    kmeans = cluster_sentence_vectors(sent_vecs, n_clusters=min(num_sentences, sent_vecs.shape[0]), normalize_vectors=True)
    return build_summary(kmeans, sent_vecs, sentences)
if __name__ == "__main__":
    # Ví dụ văn bản tiếng Việt
    text = """

Biến đổi khí hậu toàn cầu đang là thách thức lớn nhất mà nhân loại phải đối mặt trong thế kỷ 21, với những tác động ngày càng rõ rệt và nghiêm trọng. Nguyên nhân chủ yếu được xác định là do sự gia tăng nồng độ khí nhà kính, đặc biệt là CO2, phát thải từ các hoạt động công nghiệp, giao thông vận tải và nông nghiệp thâm canh. Những hậu quả nhãn tiền bao gồm nhiệt độ trung bình Trái Đất tăng lên, mực nước biển dâng do băng tan ở các cực và đỉnh núi, cùng với sự gia tăng tần suất và cường độ của các hiện tượng thời tiết cực đoan như bão lũ, hạn hán và cháy rừng.
Trước tình hình cấp bách này, việc chuyển dịch từ năng lượng hóa thạch sang năng lượng tái tạo không chỉ là một xu hướng mà là một yêu cầu bắt buộc để đảm bảo sự phát triển bền vững. Năng lượng mặt trời và năng lượng gió đang nổi lên như hai trụ cột chính của quá trình chuyển đổi này. Công nghệ pin mặt trời (photovoltaic) đã đạt được những bước tiến đáng kể trong hiệu suất và giảm chi phí sản xuất, khiến điện mặt trời trở thành một lựa chọn kinh tế cho cả quy mô hộ gia đình và các trang trại điện lớn. Tương tự, các tuabin gió ngày càng lớn hơn và hiệu quả hơn, có thể khai thác sức gió ở cả trên đất liền và ngoài khơi (offshore wind farms).
Tuy nhiên, việc tích hợp năng lượng tái tạo vào lưới điện quốc gia vẫn gặp phải nhiều rào cản kỹ thuật. Tính không ổn định và phụ thuộc vào thời tiết của năng lượng gió và mặt trời đặt ra thách thức lớn về khả năng lưu trữ điện năng. Các giải pháp lưu trữ tiên tiến, như công nghệ pin lithium-ion dung lượng cao hoặc hệ thống thủy điện tích năng, đang được nghiên cứu và phát triển mạnh mẽ để giải quyết bài toán này. Hơn nữa, sự hợp tác quốc tế, các chính sách hỗ trợ từ chính phủ (ví dụ như thuế carbon hoặc trợ cấp cho năng lượng xanh) đóng vai trò then chốt trong việc thúc đẩy đầu tư và mở rộng quy mô năng lượng sạch trên toàn cầu.
Việc giải quyết khủng hoảng khí hậu đòi hỏi nỗ lực tổng hợp từ mọi quốc gia, ngành công nghiệp và cá nhân. Mặc dù con đường phía trước còn nhiều chông gai, nhưng sự đổi mới công nghệ và cam kết chính trị mạnh mẽ đang mở ra hy vọng về một tương lai năng lượng sạch hơn, bền vững hơn cho các thế hệ mai sau.

    """

    # Đường dẫn tới model đã lưu (.kv) và file stopwords
    w2v_model_path = "wiki.vi.kv"  # model KeyedVectors đã save
    stopwords_path = "vietnamese-stopwords.txt"

    # Số câu tóm tắt muốn lấy
    num_summary_sentences = 2

    # Chạy tóm tắt
    summary = summarize_text(
        contents=text,
        w2v_path=w2v_model_path,
        stopwords_path=stopwords_path,
        num_sentences=num_summary_sentences
    )

    print("----- TÓM TẮT -----")
    print(summary)
