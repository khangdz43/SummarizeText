import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from sklearn.preprocessing import normalize


def get_data(file_path: str) -> str:
    """Read and return text from a file path (UTF-8)."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def get_stopwords(file_path_stopwords: str) -> set:
    """Load stopwords from file (one token per line) and return a set."""
    with open(file_path_stopwords, encoding="utf-8") as f:
        words = f.read().splitlines()
    return set([w.strip() for w in words if w.strip()])


def clean_text(text: str) -> str:
    """Basic normalization: lowercase, remove extra whitespace/newlines."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\n", " ")
    return " ".join(text.split())


def split_sentences(text: str) -> list:
    """Split text into sentences using nltk (fallback)."""
    if not text:
        return []
    return nltk.sent_tokenize(text)


def load_w2v_model(path: str, binary: bool = False) -> KeyedVectors:
    """Load KeyedVectors (word2vec format). Tries the given binary flag, else flips it as fallback."""
    try:
        return KeyedVectors.load_word2vec_format(path, binary=binary)
    except Exception:
        # try opposite mode
        return KeyedVectors.load_word2vec_format(path, binary=not binary)


def sentences_to_vectors(sentences: list, w2v: KeyedVectors, stopwords: set) -> np.ndarray:
    """Convert list of sentence strings to an (n_sentences, vector_size) numpy array.

    Each sentence is tokenized with ViTokenizer and averaged over in-vocab word vectors.
    """
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
    """Cluster sentence vectors with KMeans.

    If n_clusters is not provided, it uses ratio * n_sentences (min 1, max n_sentences).
    Returns fitted KMeans or None if X is empty.
    """
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
    """Pick a representative sentence closest to each cluster center and return joined summary.

    Sentences are returned in original document order (sorted by index).
    """
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


def summarize_text(contents: str, w2v_path: str = 'vi.vec', stopwords_path: str = 'MODEL/vietnamese-stopwords.txt', num_sentences: int = 3) -> str:
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
