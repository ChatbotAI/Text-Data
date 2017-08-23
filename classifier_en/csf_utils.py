# -*- coding: utf-8 -*-

import os
import re
import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler #, RobustScaler
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, ElasticNetCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier, OneVsOneClassifier

from data_serializer import DataSerializer as ds
from debug import Debug

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

#############################################################

stop_words = nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.porter.PorterStemmer()

#############################################################

def get_data_dict(data_path, print_all=False):
    """ returns dictionary like
    d['class_name'] => list of text documents """

    data = dict()

    class_names = os.listdir(data_path)

    not_readed_files_counter = 0

    for class_name in class_names:

        data[class_name] = {'doc_list': [], 'not_readed': 0}

        files = os.listdir(data_path + class_name)

        i = 0

        for fname in files:

            i += 1

            if print_all:
                print(class_name, i, ' from ', len(files))

            if '.DS_Store' == fname:
                continue

            try:
                f = open(data_path + class_name + '/' + fname,
                         'r') #,  encoding='utf-8')

                txt = str(f.read())

                data[class_name]['doc_list'].append(txt)

            except:
                print('-------------------------------------------')
                print(class_name)
                print(fname)
                data[class_name]['not_readed'] += 1
                not_readed_files_counter += 1
                Debug.print_exception_info()

    return data, not_readed_files_counter

############################################################################

def get_vector_space_model(data_dict,
                           min_doc_list_size=0,
                           make_equal_size=False,
                           print_all=False):
    """ returns matrix, labels and vectorizer """

    doc_list = []
    labels = []

    # making doc list

    for class_name in data_dict:

        dl = data_dict[class_name]['doc_list']

        if len(dl) < min_doc_list_size > 0:
            continue

        if make_equal_size is False:
            doc_list += dl
            labels += len(dl) * [class_name]

        elif min_doc_list_size > 0:
            doc_list += random.sample(dl, min_doc_list_size)
            labels += min_doc_list_size * [class_name]

    if print_all:
        print('making matrix')

    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                 tokenizer=tokenize)

    tfidf_matrix = vectorizer.fit_transform(doc_list)

    if print_all:
        print('matrix ready')

    return (tfidf_matrix,
            labels,
            vectorizer)

############################################################################

def get_vector_space_model_train_test(data_dict,
                                      min_doc_list_size=0,
                                      make_equal_size=False,
                                      train_perc=0.66):
    """ returns matrixes, labels, vectorizer """

    print('\ncreating models ...\n')

    doc_list = []
    labels = []

    print('making doc list')

    for class_name in data_dict:

        try:

            dl = data_dict[class_name]['doc_list']

            if len(dl) < min_doc_list_size > 0:
                continue

            if make_equal_size is False:
                doc_list += dl
                labels += len(dl) * [class_name]

            elif min_doc_list_size > 0:
                doc_list += random.sample(dl, min_doc_list_size)
                labels += min_doc_list_size * [class_name]

        except:
            Debug.print_exception_info()



    print('making matrix')
    doc_list_train, doc_list_test, Y_train, Y_test = train_test_split(\
                                    doc_list,
                                    labels,
                                    test_size=1-train_perc,
                                    random_state=random.randint(0,31))

    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                 tokenizer=tokenize)

    matrix_train = vectorizer.fit_transform(doc_list_train)

    matrix_test = vectorizer.transform(doc_list_test)

    print('matrix ready')

    return (matrix_train,
            matrix_test,
            Y_train,
            Y_test,
            vectorizer)

##############################################################################

def apply_scaler(matrix_train,
                 matrix_test,
                 scaler_name):

    scaler = scaler_name()

    return (scaler.fit_transform(matrix_train),
            scaler.transform(matrix_test))

##############################################################################

def reduce_dimension(matrix_train,
                     matrix_test,
                     svd_components_amount):

    vector_dim = matrix_train.shape[1]

    if svd_components_amount < vector_dim:

        svd = TruncatedSVD(n_components=svd_components_amount)

        matrix_train = svd.fit_transform(matrix_train)
        matrix_test = svd.transform(matrix_test)

        print('svd explained variance ratio: ', svd.explained_variance_ratio_)
        print(sum(svd.explained_variance_ratio_))
    return matrix_train, matrix_test, svd

#############################################################

def get_accuracy(pp,
                 Y_test,
                 clf,
                 data,
                 train_perc,
                 print_all=True):

    label_names = get_label_names()

    report = dict()

    num_to_label_dict = dict()
    label_to_num_dict = dict()
    i = 0
    for label in sorted(data):
        i += 1
        num_to_label_dict[i] = label
        label_to_num_dict[label] = i

    # documents amount for each category
    for label in data:
        report[label] = dict()
        report[label]['len'] = len(data[label]['doc_list'])

    micro = recall_score(Y_test, pp, average='micro')

    recall_by_cat = recall_score(Y_test, pp, average=None)

    acc = accuracy_score(Y_test, pp)

    i = 1
    tuples = []

    for x in recall_by_cat:

        label = num_to_label_dict[i]

        report[label]['accuracy'] = round(x, 4)

        name = label_names[label]
        cat = label + ' - ' + name
        doc_list_len = str(report[label]['len'])
        i += 1

        sx = str(round(x, 3))
        if len(sx) < 5:
            sx = sx + '0' * int(5-len(sx))

        tuples.append((cat, doc_list_len, sx))

    tuples = sorted(tuples, key=lambda x: x[2], reverse=True)

    s = "%50s: %5s\t%5s" % ('cat', 'len', 'acc')
    print()
    print(s)
    print()

    for cat, doc_list_len, accuracy in tuples:
        s = "%50s: %5s\t%5s" % (cat, doc_list_len, accuracy)
        print(s)

#    print()
#    print('acc: ', acc)
#    print('micro: ', micro)

    return report

#######################################################################

def get_classifier(name, clf_params):

    clfs = {
        'forest':RandomForestClassifier(n_estimators=clf_params[0]),

        'sgd':SGDClassifier(loss='hinge', penalty='l2',
             alpha=1e-3, n_iter=5, random_state=42),

        'logistic':LogisticRegression(C=1.0, class_weight=None,
                           dual=False, fit_intercept=True,
                           intercept_scaling=1, penalty='l2',
                           random_state=None,
                           tol=0.0001),

        'knn':KNeighborsClassifier(),
        'svm':svm.SVC(kernel='linear', C=0.01),
        'elastic':ElasticNetCV(),
        'svc':LinearSVC(),
        'gnb':GaussianNB(),
        'bnb':BernoulliNB(),
        'aggress':PassiveAggressiveClassifier(n_iter=50),
        'perceptron':Perceptron(n_iter=50),
        'ncentr':NearestCentroid(),
        'ridge':RidgeClassifier(tol=1e-2, solver="lsqr"),

        'ada':AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                 algorithm="SAMME",
                 n_estimators=200),
        'extra_trees':ExtraTreesClassifier(),
        'dtree':tree.DecisionTreeClassifier(criterion='entropy',
                                           max_depth=3,min_samples_leaf=5),
        'mnnb':MultinomialNB()
  }

    return clfs[name]

##############################################################################

def stem_tokens(tokens):

    stemmed = []

    for item in tokens:

        word = re.sub(r'[\.,!?;:]', '', item)

        if not len(re.findall(r'\d', word)) > 0 and \
           len(word) > 1 and \
           len(re.findall(r'\W', word)) == 0:

            stemmed.append(stemmer.stem(word))

    return stemmed

##########################################################################

def tokenize(text):

    tokens = nltk.word_tokenize(text)

    stems = stem_tokens(tokens)

    return stems

##########################################################################

def describe_the_data(data):

    tuples = []
    for key in data:
        tuples.append((key, len(data[key]['doc_list'])))

    tuples = sorted(tuples, key=lambda x: x[1])

    print()
    s = "%50s:\t%5s" % ("label", "num")
    print()
    x = []
    for label, num in tuples:
        x.append(num)
        s = "%50s:\t%5d" % (label, num)
        print(s)

    s = "%50s" % ('------------------------------')
    print(s)
    s = "%50s:\t%5d" % ("TOTAL", sum(x))
    print(s)
    print()
    s = "%50s:\t%5d" % ("MEAN", np.ceil(np.mean(x)))
    print(s)
    s = "%50s:\t%5d" % ("MEDIAN", np.ceil(np.median(x)))
    print(s)
    s = "%50s:\t%5d" % ("MIN", min(x))
    print(s)
    s = "%50s:\t%5d" % ("MAX", max(x))
    print(s)

##########################################################3


def get_label_names():
    s = """
Category ID	Category Name
1	Accountancy / Finance
2	Administrative / Secretarial
3	Architecture / Design
4	Arts / Culture / Writing
5	Banking / Insurance
6	Bars / Hotels / Restaurants
7	Call-Centre
8	Childcare / Caring
9	Construction / Engineering
10	Factory / Operative / Manual
11	Fitness / Leisure / Beauty
12	Graduate
13	Human Resources
14	IT / Programming / Software
15	Languages
16	Legal
17	Managers / Supervisors
18	Manufacturing / Engineering
19	Marketing / Public Relations
20	Medical & Nursing
21	Other
22	Pharmaceutical / Science
23	Public Sector
24	Recruitment
25	Retail
26	Sales
27	Security
28	Teaching / Training
29	Trades
30	Travel / Tourism
31	Voluntary / Charity
32	Warehousing / Logistics / Transport
33	Work Experience / Internship
34	Work From Home"""

    res = dict()
    lines = s.split('\n')

    for line in lines:
        label = line[:2]
        num = None
        try:
            num = int(label)
        except:
            pass
        if num==None:
            continue
        label = str(num)
        if len(label) < 2:
            label = '0' + label
        res[label] = line[2:].strip()

    return res

##################################################################













