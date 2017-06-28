import string
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
from nltk.corpus import pros_cons
from nltk.stem.lancaster import LancasterStemmer
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures



def bag_of_words(words):
    return dict([(word, True) for word in words])


def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=10):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征


# 去除掉文字的标点符号,并切分成单词列表
def preprocess(paragraph):
    # 标点符号
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    # 分词
    paragraph = nltk.word_tokenize(paragraph.lower())
    return [word for word in paragraph if not word in (stopwords.words("english") + english_punctuations)]


# 找到词语的主干部分,一般用于分类
def stem(word_list):
    st = LancasterStemmer()
    return [st.stem(word) for word in word_list]


sentence = "As the Chinese currency is not freely convertible under the capital account," \
           + " the central bank has to purchase foreign currency generated by China's trade surplus and foreign investment in the country, adding funds to the money market." \
           + "The narrowing decline indicated easing pressure from capital flight as the Chinese economy firms up and the yuan stabilizes against the U.S. dollar." \
           + "Official data showed China forex reserves climbing to 3.0295 trillion U.S. dollars at the end of April from 3.0091 trillion dollars a month earlier." \
           + "This was the first time since June 2014 the reserves expanded for three consecutive months"
# print(preprocess(sentence))
# print(stem(preprocess(sentence)))
# print(remove_punctuation(word))
nltk.download()
# print(pros_cons.readme())
# print(bigram_words(preprocess(sentence)))
# print(stopwords.words("english"))