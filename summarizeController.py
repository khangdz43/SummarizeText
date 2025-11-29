import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.models import KeyedVectors
from pyvi import ViTokenizer

# Đọc dữ liệu
def getData(filePath):
    with open(filePath, encoding='utf-8') as file:
        return file.read()

# lấy stopWord trong file
def getStopWords(filePathStopWords):
    with open(filePathStopWords, encoding='utf-8') as file:
        stopWords = file.read().split('\n')
    # loại bỏ khoảng trắng thừa
    stopWords = [sw.strip() for sw in stopWords if sw.strip()]
    return stopWords


# tiền xử lí văn bản
def preProcessing(contents):
    contents = contents.lower()
    contents = contents.replace('\n', ' ')
    contents = contents.strip()
    return contents

# tách thành câu 
def devision(sentences):
    return nltk.sent_tokenize(sentences)


# Vector hóa câu bằng Word2Vec
def sentencesVector(sentences, stopWords):
    # Load word2vec (wiki.vi.vec dạng text, binary=False)
    # w2v = KeyedVectors.load_word2vec_format("wiki.vi.vec", binary=False)
    # w2v.save("wiki.vi.kv")
    w2v = KeyedVectors.load("wiki.vi.kv")
    vocab = w2v.key_to_index # lấy tập từ vựng trong từ điển
    dim = w2v.vector_size # kích thước vector của từ
    # print(f"Kích thước từ vựng: {len(vocab)} - Kích thước vector: {dim}")
    X = []
    # từng câu trong đoạn
    for sent in sentences:
        # tokenize câu
        sent = ViTokenizer.tokenize(sent)
        # cơ_quan điều_tra xác_định từ năm 2022 đến cuối năm 2024 , t . q . t ( sn 1989 , trú đà_nẵng ) đã làm giả căn_cước công_dân để đăng_ký thành_lập 187 doanh_nghiệp và làm giả nhiều con_dấu để mở hơn 600 tài_khoản doanh_nghiệp , bán cho các đối_tượng tội_phạm
        # print("==========" + sent)
        words = sent.split(" ")

        # tính vector câu CHƯA HIỂU ĐOẠN NÀY
        sent_vector = np.zeros(dim)
       

        num_words = 0
        for word in words:
            # kiểm tra stopword
            if word in vocab and word not in stopWords:
                sent_vector += w2v[word]
                num_words += 1
        if num_words > 0:
            sent_vector = sent_vector / num_words
        X.append(sent_vector)
    return np.array(X)


# Phân cụm KMeans
def sentencesCluster(X,scale):
    n_clusters = max(1, len(X) * scale // 100)  # ít nhất 1 cluster
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X)
    return kmeans


# Tạo summary
def buildSummary(kmeans, X, sentences, scale):
    n_clusters = max(1, len(X) * scale // 100)
    # Lấy câu gần tâm cụm
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

    # Tính vị trí trung bình câu trong từng cluster
    avg = []
    for i in range(n_clusters):
        idx = np.where(kmeans.labels_ == i)[0]
        avg.append(np.mean(idx))

    # Sắp xếp cụm theo vị trí câu
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])

    # Ghép câu tạo summary
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    return summary


# Hàm tóm tắt chính
def summarization(contents,scale):
    stop_words = getStopWords("vietnamese-stopwords.txt")
    contents = preProcessing(contents)
    sentences = devision(contents)
    X = sentencesVector(sentences, stop_words)
    kmeans = sentencesCluster(X, scale)
    summary = buildSummary(kmeans, X, sentences , scale)
    return summary


# TEST
# if __name__ == "__main__":
#     text = """
# Vào khoảng 15h ngày 26/11, anh Chung, 45 tuổi, đang ở chỗ làm thì nhận được cuộc gọi của vợ. Cô hốt hoảng thông báo rằng tòa nhà trong cụm chung cư Wang Fuk Court ở khu Đại Bộ, Hong Kong bị cháy, nhưng cô và mèo cưng không thể thoát được ra ngoài. Anh Chung vội vã trở về và thấy tòa nhà 31 tầng đang bốc cháy, với những đám khói đen cuồn cuộn bốc lên. Vụ hỏa hoạn kéo dài suốt một ngày, thiêu rụi 7 tòa nhà, trong đó có nơi vợ chồng anh Chung ở. Cảnh sát và lính cứu hỏa khi đó đã cấm đường, lập vành đai phong tỏa, ngăn mọi người tiếp cận những tòa nhà đang cháy. Anh Chung chỉ có thể bất lực đứng bên ngoài theo dõi nỗ lực cứu hộ và cố gắng giữ liên lạc với vợ. Suốt đêm 26/11, anh Chung đi khắp nơi để hỏi lính cứu hỏa về tình hình bên trong, nhưng đều không nhận được câu trả lời. Anh liên tục gọi điện cho vợ, với cảm xúc lo lắng, sợ hãi đan xen. Người vợ nói với anh Chung rằng có lẽ cô sắp ngất khi đám khói ngày càng dày đặc. Trong lời cuối nhắn nhủ tới vợ, anh Chung chỉ biết động viên cô: "Đừng bỏ cuộc em nhé". "Có lẽ cô ấy đã ngất đi. Tôi không dám gọi cho cô ấy nữa", anh Chung kể với đôi mắt đỏ hoe. Nhiều giờ trôi qua kể từ cuộc gọi cuối, anh đã chuẩn bị tinh thần cho điều tồi tệ nhất. "Cô ấy nếu có ra đi cũng đi cùng với con mèo của chúng tôi, con mèo mà cô ấy rất yêu quý", anh bật khóc nói. Gia đình anh Chung chuyển đến tòa Wang Cheong cách đây một thập kỷ. Đây là tòa nhà bốc cháy đầu tiên trong cụm chung cư Wang Fuk Court. Anh Chung cho biết khi đám cháy bùng phát, khói dày đặc nhanh chóng bủa vây tầng 23, nơi gia đình anh sinh sống, khiến vợ anh không thể tìm được đường thoát. Đám cháy khiến ít nhất 128 người thiệt mạng và gần 300 người chưa được tìm thấy. Hy vọng về những người mất tích ngày càng mong manh, khi lính cứu hỏa rà soát từng căn hộ mà không phát hiện người sống sót. Theo điều tra dân số năm 2021, gần 40% cư dân của khu Wang Fuk Court ở độ tuổi từ 65 trở lên. Đó là lý do nhiều người lo ngại còn rất nhiều cư dân mắc kẹt, do họ đã lớn tuổi và khó chạy thoát nhanh. Cô Fung, 40 tuổi, vừa cùng bố mẹ chuyển tới khu nhà này vào năm ngoái, vẫn chưa tìm thấy thông tin của mẹ. Trong lúc đám cháy bùng phát, cô và bố đi làm, chỉ có mẹ ở nhà. Cô Fung sau đó nhận được cuộc gọi từ hàng xóm nói rằng đang cùng mẹ của cô nấp trong nhà vệ sinh. Nhưng tới nửa đêm 26/11, cô mất liên lạc với người hàng xóm này. Cô vẫn nuôi hy vọng mong rằng sẽ thấy mẹ được giải cứu. Nhiều người dân Hong Kong vẫn mong chờ phép màu sẽ xảy đến với người thân còn mắc kẹt trong cụm chung cư, khi chính quyền khẳng định vẫn nỗ lực tìm kiếm người còn sống. "Tôi muốn cứu vợ, dù cô ấy còn sống hay đã mất", anh Chung nói.
#     """

#     print("----- TÓM TẮT -----")
#     print(summarization(text))
