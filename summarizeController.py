import nltk
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.models import KeyedVectors
from pyvi import ViTokenizer

def getData(filePath):
    with open (filePath, encoding='utf-8') as file:
        data = file.read()
    return data

def getStopWords(filePathStopWords):
    with open (filePathStopWords, encoding='utf-8') as file:
        stopWords = file.read()
    stopWords = stopWords.split('\n')
    return stopWords
def preProcessing(contents):
    contents = contents.lower()
    # loai bo kí tự thừa
    contents = contents.replace('\n', ' ')
    # bỏ khoảng trắng thừa
    contents = contents.strip()
    return contents
def devision(sentences):
    sent_tokens = nltk.sent_tokenize(sentences)
    return sent_tokens

def sentencesVector(sentences , stopWords):
    # timf vecto cho tung tu bang word2vec
    w2v = KeyedVectors.load_word2vec_format('vi.vec', binary=True)
    # lấy danh sách các từ trong từ điển
    vocab = w2v.wv.vocab
    # khời tạo list lưu các vecto đại diện cho từng câu
    X = []
    for sent in sentences:
        sent = ViTokenizer.tokenize(sent)
        words = sent.split(" ")
        # khởi tạo vecto 100 chiều
        sent_vector = np.zeros(100)
        num_words = 0
        for word in words:
            if word in vocab and word not in stopWords:
                sent_vector += w2v[word]
                num_words += 1
        X.append(sent_vector / num_words)
    return X
def sentencesCluster(X):
    n_clusters = len(X) *30 //100


    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans
def buildSumary( kmeans ,X , sentences):
    n_clusters = len(X) *30 //100
    avg = []
    for i in range(n_clusters):
        idx = np.where(kmeans.labels_ == i)[0]
        avg.append(np.mean(idx))
        closet , _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary = ' '.join([sentences[closet[idx]] for idx in ordering])
        return summary
    def summarization(contents):
        filePathStopWords = 'MODEL/vietnamese-stopwords.txt'
        stop_words = getStopWords(filePathStopWords)
        contents = preProcessing(contents)
        sentences = devision(contents)
        X = sentencesVector(sentences, stop_words)
        kmeans = sentencesCluster(X)
        summary = buildSumary(kmeans, X, sentences)
        print("-----Tóm tắt-----")
        return summary
    