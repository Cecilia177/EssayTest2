from stanfordcorenlp import StanfordCoreNLP
from sentence import Sentence
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
    for n in nps:
        total_len += len(n.split(" ")) if sentence.flag == "en" else len(n.replace(" ", ""))
    return float(total_len) / len(nps) if len(nps) != 0 else 0


def get_phrases_ratio(phrase, refs, answer, model):
    ref_phrase = 0
    for f in refs:
        ref_phrase += get_phrases(phrase, f, model)
    ref_phrase /= len(refs)
    return get_phrases(phrase, answer, model) / ref_phrase


# if __name__ == '__main__':
#     zh_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27", lang='zh')
#     en_model = StanfordCoreNLP(r"H:\Download\stanford-corenlp-full-2018-02-27")
#     s0 = "xu shijie's stories about how to learn english, how to help others."
#     s1 = "我们不必学会如何让心灵健康，我们的心灵就像我们的身体一样"
#     ss0 = Sentence(text=s0, language="en")
#     ss1 = Sentence(text=s1, language="ch")
#     np0 = get_phrases(phrase='NP', sentence=ss0, model=en_model)
#     np1 = get_phrases(phrase="NP", sentence=ss1, model=zh_model)
#     # VP0 = get_phrases(phrase='VP', sentence=ss0, model=en_model, language="en")
#     print(np0)
#     print(np1)

