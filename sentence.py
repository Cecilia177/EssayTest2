import math
import string
import zhon.hanzi as hz
import re
import jieba
import jieba.posseg as psg


class Sentence:
    """
    Attributes:
        text -- The original text of the sentence. Updated in method self.remove_punc()
        flag -- "ch" or "en", defines the language.
        seg_length -- number of phrases after segmentation.  Updated in method self.segment()
        pure_text -- made of phrases and blank spaces after removing punctuations. Updated in method self.segment()
        ngram -- A dict, the key of which is 1~4gram string and the value is the count. Updated in method self.get_ngram().
    """
    def __init__(self, text, language):
        self.original = text
        self.text = text
        self.ngram = {}
        self.seg_length = 0
        self.flag = language
        self.pure_text = ""
        self.part_of_speech = []
        self.speeches_gram = {}

    def remove_punc(self):
        """
        Replace all punctuations with "sss" in self.text.

        """
        # for c in string.punctuation + hz.punctuation:
        #     self.text = self.text.replace(c, "<sss>")
        self.text = re.sub("[{}]+".format(hz.punctuation + string.punctuation), "<sss>", self.text)

    def segment(self):
        """
        Get segmentation of this paragraph.
        Set self.pure_text and self.seg_length
        """
        sub_sentences = self.text.split("<sss>")
        new_text = "</s> </s> </s> "
        length = 0
        for s in sub_sentences:
            if s:
                segments = " ".join(jieba.cut(s)) if self.flag == "ch" else s
                length += len([i for i in segments.split(" ") if i != ''])
                new_text += segments + " </s> </s> </s> "
        self.text = new_text[:-1]
        pure_text = self.text.replace(" </s> </s> </s>", "")
        pure_text = pure_text.replace("</s> </s> </s> ", "")
        self.pure_text = pure_text            # pure_text is made of phrases and blank spaces
        self.seg_length = length              # seg_length is the number of phrases instead of string length

    def get_ngram(self):
        """
        Set self.ngram

        """
        words_list = [i for i in self.text.split(" ") if i]
        # print("wordslist:", words_list)
        ngram = {}       # key is the actual 1~4gram sequence and value is the number of it
        for i in range(4):
            for j in range(len(words_list)):
                if j + i >= len(words_list):
                    break
                gramkey = words_list[j]
                index = 0
                while index < i:
                    gramkey += " " + words_list[j + index + 1]
                    index += 1
                if gramkey not in ["</s>", "</s> </s>", "</s> </s> </s>"]:
                    if gramkey not in ngram.keys():
                        ngram[gramkey] = 1
                    else:
                        ngram[gramkey] += 1
        self.ngram = ngram

    def get_part_of_speech(self):
        """
        Segment the original text, get part-of-speeches.
        """
        text = self.original.replace(".", "。")
        speeches = psg.cut(text)
        speech_list = []
        # print(len(list(speeches)))
        for p in speeches:
            if p.flag == 'x':
                length = len(speech_list)
                speech_list[length: length] = ['/x', '/x', '/x', '/x']
            else:
                speech_list.append("/" + p.flag)

        if speech_list[-1] != '/x':
            length = len(speech_list)
            speech_list[length: length] = ['/x', '/x', '/x', '/x']
        speech_list[0:0] = ['/x', '/x', '/x', '/x']
        speeches_gram = {}
        for index, p in enumerate(speech_list):
            if index + 5 > len(speech_list):
                break
            key = " ".join(speech_list[index:index + 5])
            if key in speeches_gram.keys():
                speeches_gram[key] += 1
            else:
                speeches_gram[key] = 1
        self.speeches_gram = speeches_gram

    def preprocess(self):
        self.remove_punc()
        # print("removed punc:", self.text)
        self.segment()
        # print("segmented:", self.text)
        self.get_ngram()


# if __name__ == '__main__':
#     ss = "Xu Shijie learned, English by himself and he introduced Chinese culture to foreign tourists."
#     ss1 = "正如你将意识到的,知道心理健康一直有效并且知道去信任它,这将让我们放缓时光，快乐地生活"
#     s = Sentence(text=ss, language="en")
#     s1 = Sentence(text=ss1, language="ch")
#     s.preprocess()
#     s1.preprocess()
#     print(s.pure_text)
#     print(s.text)
#     print(s.seg_length)
#     print(s.ngram)

