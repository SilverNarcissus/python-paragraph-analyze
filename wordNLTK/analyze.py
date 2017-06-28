import collections
import nltk.classify.util, nltk.metrics
import pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import pros_cons
from nltk.metrics import precision
from nltk.metrics import recall
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
from nltk.corpus import pros_cons
from nltk.stem.lancaster import LancasterStemmer
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# 标记函数
# 二元词组 + 单词特征
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    if (len(words) <= 3):
        return dict([(ngram, True) for ngram in words])
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    # 这里必须用chain，因为传入的 word 可能只是 iterable 不是list
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


# 单词特征
def word_feats(words):
    return dict([(word, True) for word in words])


# 被过滤的单词特征
def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopwords.words("english")])


############################

def review_evaluate_classifier(featx):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    print(movie_reviews.words(fileids=[negids[0]]))
    negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = int(len(negfeats) * 3 / 4)
    poscutoff = int(len(posfeats) * 3 / 4)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

    classifier = NaiveBayesClassifier.train(trainfeats)
    pickle.dump(classifier, open('/Users/SilverNarcissus/PycharmProjects/wordAnalysis/reviews_classifier.pkl', 'wb'))
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        # 分类化过程
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    print('pos precision:', precision(refsets['pos'], testsets['pos']))
    print('pos recall:', recall(refsets['pos'], testsets['pos']))
    print('neg precision:', precision(refsets['neg'], testsets['neg']))
    print('neg recall:', recall(refsets['neg'], testsets['neg']))
    classifier.show_most_informative_features()


def pros_cons_evaluate_classifier(featx):
    print(pros_cons.sents(categories='Pros')[1])
    posfeats = [(featx(f), 'pros') for f in pros_cons.sents(categories='Pros')]
    negfeats = [(featx(f), 'cons') for f in pros_cons.sents(categories='Cons')]

    negcutoff = int(len(negfeats) * 3 / 4)
    poscutoff = int(len(posfeats) * 3 / 4)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

    # pipeline = Pipeline([('tfidf', TfidfTransformer()),
    #                      ('chi2', SelectKBest(chi2, k=1000)),
    #                      ('nb', MultinomialNB())])
    # classifier = SklearnClassifier(pipeline)
    classifier = NaiveBayesClassifier.train(trainfeats)
    # pickle.dump(classifier, open('/Users/SilverNarcissus/PycharmProjects/wordAnalysis/pros_cons_classifier.pkl', 'wb'))
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        # 分类化过程
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    print('pos precision:', precision(refsets['pros'], testsets['pros']))
    print('pos recall:', recall(refsets['pros'], testsets['pros']))
    print('neg precision:', precision(refsets['cons'], testsets['cons']))
    print('neg recall:', recall(refsets['cons'], testsets['cons']))
    classifier.show_most_informative_features()


def preprocess(paragraph):
    # 标点符号
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    # 分词
    paragraph = nltk.word_tokenize(paragraph.lower())
    return [word for word in paragraph if not word in (stopwords.words("english") + english_punctuations)]


# pros_cons 分类
def pros_cons_classify(featx, words):
    classifier = pickle.load(open('/Users/SilverNarcissus/PycharmProjects/wordAnalysis/pros_cons_classifier.pkl', 'rb'))
    return classifier.classify(featx(words))


#review_evaluate_classifier(bigram_word_feats)
# pros_cons.sents(categories='Cons')
#pros_cons_evaluate_classifier(bigram_word_feats)
# print(pros_cons_classify(bigram_word_feats, preprocess("As the Chinese currency is not freely convertible under the capital account," \
#            + " the central bank has to purchase foreign currency generated by China's trade surplus and foreign investment in the country, adding funds to the money market." \
#            + "The narrowing decline indicated easing pressure from capital flight as the Chinese economy firms up and the yuan stabilizes against the U.S. dollar." \
#            + "Official data showed China forex reserves climbing to 3.0295 trillion U.S. dollars at the end of April from 3.0091 trillion dollars a month earlier." \
#            + "This was the first time since June 2014 the reserves expanded for three consecutive months")))
# # print(pros_cons.__getattr__())
#review_evaluate_classifier(bigram_word_feats)