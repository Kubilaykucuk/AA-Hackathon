import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Turkish stopwords
stop_words = set(stopwords.words('turkish'))

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in word_tokenize(sent1) if w not in stopwords]
    sent2 = [w.lower() for w in word_tokenize(sent2) if w not in stopwords]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_similarity([vector1], [vector2])[0][0]

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

def textrank(text, stopwords=stop_words):
    sentences = sent_tokenize(text)
    
    similarity_matrix = build_similarity_matrix(sentences, stopwords)
    
    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return ranked_sentences

# Example text in Turkish
text = """
Yeni uygulamayla halk sağlığının yanı sıra çevreye yönelik de önlemler alındığı aktarılan açıklamada,
geçen yıl her hafta yaklaşık 5 milyon tek kullanımlık elektronik sigaranın çöpe atıldığı, bu sayının
2022'de haftalık 1,3 milyon olduğu belirtildi.
Çöpe atılan 5 milyon tek kullanımlık elektronik sigaranın içinde bulunan lityum bataryaların 5 bin
elektrikli araç bataryasına eşit olduğunun altı çizilen açıklamada, Başbakan Rishi Sunak ile Sağlık ve
Çevre Bakanlarının değerlendirmelerine de yer verildi.
Sunak, elektronik sigaraların insan sağlığına uzun vadede etkilerinin bilinmediğine dikkati çekerek,
"Elektronik sigaralar, sigarayı bırakmada kullanışlı olsa da çocuklara satışı kabul edilemez. Başbakan
olarak ülkemiz için uzun vadede doğru olanı yapmak görevim. Bu nedenle tek kullanımlık sigaralara
karşı adım atıyorum." ifadelerini kullandı.
Sağlık Bakanı Victoria Atkins, "Sigara, İngiltere'de yaşanan ölümler arasında en büyük önlenebilir
sebep. Neredeyse her dakikada bir kişi sigarayla bağlantılı bir sebepten hastanelere başvuruyor. Bu,
toplumumuza her yıl 17 milyar sterline (yaklaşık 655 milyar lira) mal olurken, sağlık sistemimize de
büyük yük getiriyor." değerlendirmesini yaptı.
Çevre Bakanı Steve Barclay, geri dönüşümü çok zor olan tek kullanımlık elektronik sigaraların
milyonlarcasının çöpe atıldığına dikkati çekti.
İngiltere hükümeti, geçen sene açıkladığı kararlarla 2009 sonrası doğanların hayatları boyunca tütün
ve elektronik sigara ürünleri alamamasını sağlamak için sigara alma yaşının her yıl artırılması da dahil
bir dizi önlem almıştı. Bu kapsamda "ilk sigarasız nesil" hedefini açıklayan hükümet, gelecek 5 yılda sigara kaçakçılığını
önlemek, satış standartlarını düzenlemek ve satışları denetlemek için görev gücü kurmak amacıyla
100 milyon sterlinlik bir fon oluşturulacağını duyurmuştu.

"""

ranked_sentences = textrank(text)

for idx, (score, sentence) in enumerate(ranked_sentences):
    print(f"Rank: {idx+1}, Score: {score:.4f}, Sentence: {sentence}")
