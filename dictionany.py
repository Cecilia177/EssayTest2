import numpy as np
import re
from sentence import Sentence
from gensim.models import KeyedVectors


def get_dict():
    with open("./files/dictionary.txt", "r", encoding='utf-8-sig') as f:
        item = f.readline()
        dic = {}
        count = 0
        key = None
        v = []
        while item:
            item = re.sub(r"[a-z]+\.", "", item)
            item = [i for i in re.split(r"\s+", item) if i]
            pattern = re.compile("[a-zA-Z]+")
            # print(item)
            if pattern.match(item[0]):
                if key is not None:
                    dic[key] = v
                    # print(key, ":", v)
                count += 1
                key = item[0]
                v = []
                for i in range(1, len(item)):
                    for j in item[i].split("，"):
                        if u'\u4e00' <= j[0] <= u'\u9fa5':
                            v.append(j)
                        elif pattern.match(j):
                            key += " " + j
            elif u'\u4e00' <= item[0][0] <= u'\u9fa5':
                for j in item[0].split("，"):
                    v.append(j)
            item = f.readline()
        if len(v) > 0:
            dic[key] = v
            # print(key, ":", v)
        return dic


def get_dict_match(original, trans, dict, vecs):
    """
    Parameters:
        original: class Sentence, ngrams property available
        trans: class Sentence, ngrams property available
    """
    original_words = [i for i in original.pure_text.split(" ") if i]
    trans_words = [i for i in trans.pure_text.split(" ") if i]
    score = 0
    print(original_words)
    print(trans_words)
    for w in original_words:
        if w not in dict.keys():
            continue
        meaning = dict[w]
        print(w, " ", meaning)
        best = float('Inf')
        best_word = ""
        for m in meaning:
            if m not in vecs.vocab:
                continue
            v1 = vecs[m]
            for x in trans_words:
                if x not in vecs.vocab:
                    continue
                v2 = vecs[x]
                if np.linalg.norm(v1 - v2) < best:
                    best = np.linalg.norm(v1 - v2)
                    best_word = x
        score += best
        print(original_words, " to remove ", w)
        original_words.remove(w)
        print(trans_words, " to remove ", best_word)
        if best_word in trans_words:
            trans_words.remove(best_word)
    return score


if __name__ == '__main__':
    source = "we do not have to learn how to be mentally healthy; it is built into us" \
               " in the same way that our bodies know how to heal a cut or mend a broken bone."
    trans = "我们不必学习如何变得心灵健康,这就跟我们身体知道如何愈合一道小伤或是治疗断骨一样自然天成."
    source_s = Sentence(source, language="en")
    trans_s = Sentence(trans, language="ch")
    source_s.preprocess()
    trans_s.preprocess()
    word_vectors = KeyedVectors.load("./model/vectors.kv", mmap='r')
    print(get_dict_match(original=source_s, trans=trans_s, dict=get_dict(), vecs=word_vectors))

