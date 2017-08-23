# -*- coding: utf-8 -*-

import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

from csf_utils import get_data_dict, \
                      get_vector_space_model, \
                      get_classifier, \
                      get_label_names
from debug import Debug
from data_serializer import DataSerializer as ds

#################################################################

class Classifier:

    def __init__(self,
                 txt_data_path, # text data
                 clf_data_path, # classifier files
                 svd_dim=1000,  # result matrix dimension
                 clf_name='svc'): # classifier name

        self.clf_name = clf_name
        self.svd_dim = svd_dim
        self.txt_data_path = txt_data_path
        self.clf_data_path = clf_data_path

        # must ends with slash
        c = self.txt_data_path[-1]
        if c != '/' or c != '\\':
            self.txt_data_path += '/'
        c = self.clf_data_path[-1]
        if c != '/' or c != '\\':
            self.clf_data_path += '/'

        self.clf = None
        self.vectorizer = None
        self.svd = None

        self.init_model_from_files()

            # for first time use relearn to create all needed files

    ##########################################################

    def classify(self, doc_list):


        # if not all classifier files exist
        if not any([self.vectorizer,
                    self.svd,
                    self.clf]):
            self.init_model_from_files()

        if not any([self.vectorizer,
                    self.svd,
                    self.clf]):
               print('classifier does not exists - use relearn')
               return None

        if isinstance(doc_list, str):
            doc_list = [doc_list]

        # transform document list to number matrix
        doc_vect = self.vectorizer.transform(doc_list)
        doc_vect = self.svd.transform(doc_vect)

        # predict the class (label)
        label = self.clf.predict(doc_vect)[0]

        num_to_label_dict = get_label_names()

        return label, num_to_label_dict[label]

    ##########################################################


    def relearn(self):

        try:

            print('getting data ... ')

            data, k = get_data_dict(self.txt_data_path, print_all=False)

            print('creating the model ...')

            tfidf_matrix, labels, self.vectorizer = \
                get_vector_space_model(data,
                                       min_doc_list_size=0,
                                       make_equal_size=False)

            print('saving vectorizer to disk ...')
            ds.serialize(self.vectorizer, self.clf_data_path + 'vectorizer')

            vector_dim = tfidf_matrix.shape[1]

            if self.svd_dim < vector_dim:

                print('reducing dimension')

                self.svd = TruncatedSVD(n_components=self.svd_dim)

                lsi_matrix = self.svd.fit_transform(tfidf_matrix)

                print('saving svd transformer to disk ...')
                ds.serialize(self.svd, self.clf_data_path + 'svd')


                clf1 = get_classifier(self.clf_name, clf_params=(10,))

                self.clf = OneVsRestClassifier(clf1)

                print('\nfitting classifyer ...')

                self.clf.fit(lsi_matrix, labels)

                print('saving classifier to disk ...')
                ds.serialize(self.clf, self.clf_data_path + 'clf')

                scores = cross_val_score(self.clf,
                                         lsi_matrix,
                                         labels,
                                         cv=5)

                print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), \
                                                       scores.std() * 2))

                print('relearning done')

        except:
            print('\nrelearning failed\n')
            Debug.print_exception_info()

    ##########################################################

    def init_model_from_files(self):

        try:
            self.clf = ds.deserialize(self.clf_data_path + 'clf')
            self.vectorizer = ds.deserialize(self.clf_data_path + 'vectorizer')
            self.svd = ds.deserialize(self.clf_data_path + 'svd')
        except:
            pass