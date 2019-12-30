import gensim
import jieba
import numpy as np
from scipy.linalg import norm
from sentence import Sentence
import jieba.posseg as psg
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import random


def vector_similarity(id1, id2, vecs, stopwords, tf_idf, keys, language):
    """
    Calculate two sentences
    Parameters:
        id1, id2: Integer, sentence id (column index).
        vecs: word_vectors file.
        stopwords: a list containing stopwords to ignore when getting the sentence vector.
        tf_idf: tf-idf matrix
        keys: sorted words list of all docs
    Return:
        A float, as the similarity of Sentence id1 and id2.
    """

    def sentence_vector_with_tf(keys, tf):
        v = np.zeros(64) if language == "ch" else np.zeros(300)
        for i, t in enumerate(tf):
            if t != 0:
                word = keys[i]
                # print(word, ":", t)
                if word not in stopwords and word in vecs.vocab:
                    v += t * vecs[word]
        return v

    tf1 = tf_idf[:, id1]
    tf2 = tf_idf[:, id2]
    v1, v2 = sentence_vector_with_tf(keys=keys, tf=tf1), sentence_vector_with_tf(keys=keys, tf=tf2)
    eu_dist = np.linalg.norm(v1 - v2)
    # cosine_dist = np.dot(v1, v2) / (norm(v1) * norm(v2)) if (norm(v1) * norm(v2)) != 0 else 0
    return eu_dist


def get_min_WCD(id, vecs, stopwords, tf_idf, keys, ref_num, language):
    min_vecsim = vector_similarity(id1=0, id2=id, vecs=vecs, stopwords=stopwords,
                                   tf_idf=tf_idf, keys=keys, language=language)
    for ref_index in range(ref_num):
        min_vecsim = min(min_vecsim, vector_similarity(id1=ref_index, id2=id, vecs=vecs,
                                                       stopwords=stopwords, tf_idf=tf_idf,
                                                       keys=keys, language=language))
    return min_vecsim


def get_min_WMD(refs, sentence, vectors):
    sent = sentence.pure_text.split(" ")
    min_wmd = vectors.wmdistance(refs[0].pure_text.split(" "), sent)
    for ref in refs:
        r = ref.pure_text.split(" ")
        min_wmd = min(min_wmd, vectors.wmdistance(r, sent))
    return min_wmd


if __name__ == '__main__':
    # model_path = "H:\\Download\\news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
    # model_path = "M:\\DATA\\baike_26g_news_13g_novel_229g.bin"
    # model_path = "M:\\DATA\\model11\\GoogleNews-vectors-negative300-SLIM.bin.gz"

    # print("-----start loading model---------")

    # model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    # fname = get_tmpfile("vectors.kv")
    # print(fname)
    # model.save(fname)
    # print("finished loading model..")
    # load local word vectors
    word_vectors = KeyedVectors.load("./model/vectors_en.kv", mmap='r')
    print(type(word_vectors))
    # with open('C:\\Users\\Cecilia\\Desktop\\stopwords.txt', 'r+') as f:
    #     stopwords = f.read().split("\n")

    # tfidf = np.loadtxt("C:\\Users\\Cecilia\\Desktop\\tfidf.txt")
    # keys = np.loadtxt("C:\\Users\\Cecilia\\Desktop\\keys.txt", dtype=str, delimiter='/n')





