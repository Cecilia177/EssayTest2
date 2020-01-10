from stanfordcorenlp import StanfordCoreNLP
from sentence import Sentence
import re
from collections import defaultdict
import json
import re


def get_phrases(phrase, sentence, model):
    """
    Parameter:
        phrase: 'NP' or 'VP'
        sentence: class Sentence
        model: parsing model
    Return:
        float, average length of phrases.
    """
    parsed_ser = model.parse(sentence.original)
    print(parsed_ser)
    flag = 0
    nps = []
    count = 0
    total_len = 0
    for i in range(len(parsed_ser) - 1):
        if parsed_ser[i: i+2] == phrase and flag == 0:
            flag = 1
            temp = ""
            count += 1
        elif flag == 1:
            # if language == "ch" and u'\u4e00' <= parsed_ser[i] <= u'\u9fa5':
            #     temp += parsed_ser[i]
            if 'a' <= parsed_ser[i] <= 'z' or u'\u4e00' <= parsed_ser[i] <= u'\u9fa5' or parsed_ser[i] == " ":
                temp += parsed_ser[i]
                # print(temp)
            elif parsed_ser[i] == '(':
                count += 1
            elif parsed_ser[i] == ')':
                count -= 1
                if count == 0:
                    flag = 0
                    temp = re.sub(r"\s\s+", " ", temp)
                    temp = temp.strip()
                    nps.append(temp)
    print(nps)
    for n in nps:
        total_len += len(n.split(" ")) if sentence.flag == "en" else len(n.replace(" ", ""))
    return float(total_len) / len(nps) if len(nps) != 0 else 0


def get_phrases_ratio(phrase, refs, answer, model):
    ref_phrase = 0
    for f in refs:
        ref_phrase += get_phrases(phrase, f, model)
    ref_phrase /= len(refs)
    return get_phrases(phrase, answer, model) / ref_phrase


def get_structure(sentence, model, stopwords):
    parsing = model.parse(sentence.original)
    tokens = re.sub(r"\(*\)*", "", parsing)
    tokens = tokens.replace('\r\n', '')
    tokens = [t for t in tokens.split(" ") if t]
    parents = []
    pattern = re.compile("[A-Z]+")
    parents_list = {}
    for t in tokens:
        if pattern.match(t):
            parents.append(t)
        else:
            if t not in stopwords:
                parents_list[t] = parents.copy()
            parents.pop()
    return parents_list


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = {}
        self.parent = None
        self.flag = "pof"


def tree():
    return defaultdict(tree)


def getdics(parents, leaf, tree, pos):
    if pos == len(parents):
        return leaf
    else:
        tree[parents[pos]] = getdics(parents, leaf, tree, pos+1)
        return tree


if __name__ == '__main__':
    # zh_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27", lang='zh')
    # s1 = "我们不必学会如何让心灵健康，我们的心灵就像我们的身体一样"
    # ss1 = Sentence(text=s1, language="ch")
    # np1 = get_phrases(phrase="NP", sentence=ss1, model=zh_model)

    en_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27")
    s0 = "the girl sing into a microphone"
    s1 = "the girl sing a song"
    ss0 = Sentence(text=s0, language="en")
    ss1 = Sentence(text=s1, language="en")

    with open("./files/stopwords_en.txt", 'r') as f:
        stopwords = f.read().split("\n")

    stru1 = get_structure(ss0, model=en_model, stopwords=stopwords)
    stru2 = get_structure(ss1, model=en_model, stopwords=stopwords)
    print(stru1)
    print(stru2)
    for k in stru1.keys():
        if k in stru2.keys():
            count = 0
            for i in range(0, min(len(stru1), len(stru2))):
                if stru1[k][i] == stru2[k][i]:
                    count += 1
                else:
                    break









