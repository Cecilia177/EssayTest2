import numpy as np
import re
from sentence import Sentence
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer


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


def get_key_match(original, trans, vecs):
    """
    Parameters:
        original: class Sentence, keywords property available
        trans: class Sentence, keywords property available
    """
    # original_words = [i for i in original.pure_text.split(" ") if i]
    # trans_words = [i for i in trans.pure_text.split(" ") if i]
    score = 0
    if trans.seg_length == 0:
        return score
    pairs = {}
    matched = []
    dict = get_dict()
    for w in original.keywords:
        if w not in dict.keys():
            continue
        meaning = dict[w]
        best = -float('Inf')
        best_word = ""
        for m in meaning:
            if m not in vecs.vocab:
                continue
            # v1 = vecs[m]
            for x in trans.keywords:
                if x not in vecs.vocab:
                    continue
                # v2 = vecs[x]
                if vecs.similarity(m, x) > best:
                    # best = np.linalg.norm(v1 - v2)
                    best = vecs.similarity(m, x)
                    best_word = x

        if best_word != "":
            score += best
            if w not in pairs.keys():
                pairs[w] = best_word
            else:
                pairs[w] += " " + best_word
            matched.append(best_word)
            print("pairs[" + w + "]: " + pairs[w])
    return score / len(pairs.keys())


if __name__ == '__main__':
    source = "we do not have to learn how to be mentally healthy; it is built into us" \
               " in the same way that our bodies know how to heal a cut or mend a broken bone."
    trans = "我们不必一定去学习如何做到心理健康，这种能力植根于我们自身，就像我们的身体知道如何愈合伤口，如何修复断骨"
    # trans = "我们没有必要去学习怎样保持心理健康;它们总是通过相同的方式在我们体内形成。"
    source = Sentence(source, language="en")
    trans = Sentence(trans, language="ch")
    source.preprocess()
    trans.preprocess()
    word_vectors = KeyedVectors.load("./model/vectors_128.kv", mmap='r')
    # print("健康的：", word_vectors["健康的"])
    print(get_key_match(original=source, trans=trans, dict=get_dict(), vecs=word_vectors))

