
import time
import os

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

from sklearn.multiclass import OneVsOneClassifier, \
                               OutputCodeClassifier, \
                               OneVsRestClassifier


from csf_utils import get_vector_space_model_train_test, \
                      get_classifier,  \
                      reduce_dimension, \
                      get_data_dict, \
                      get_accuracy

from data_serializer import DataSerializer as ds
from debug import Debug

##############################################################################

tt = time.time()

# directory for classifier's files
clf_data_path = '/home/yuriy/data/pyhk/report/'

# text data directory
txt_data_path = '/home/yuriy/data/pyhk/txt/'

min_doc_list_size = 0
clf_name = 'svc'
meta_name = 'ovr'
svd_dim = 100
train_perc = 0.66
trees_amount = 10
class_sample_size = 0
make_equal_size = False

# switch to True for next using
serialized_data = False
serialized_model = False
serialized_svd = False

##############################################################################

data_path = clf_data_path

# GET DATA

if not serialized_data:

    data, not_readed_files_counter = get_data_dict(txt_data_path)

    ds.serialize((data, not_readed_files_counter),
                 data_path + 'data')

else:
    try:
        data, not_readed_files_counter = ds.deserialize(data_path + \
                                                        'data')
    except:
        Debug.print_exception_info()


# CREATE MODEL

if not serialized_model:

    matrix_train, matrix_test, Y_train, Y_test, vect = \
        \
        get_vector_space_model_train_test(data,
                                          min_doc_list_size,
                                          make_equal_size,
                                          train_perc)

    ds.serialize(vect, data_path + 'vectorizer')

    ds.serialize((matrix_train, matrix_test, Y_train, Y_test),
                 data_path + 'tfidf_matrix')

else:
    try:
        matrix_train, matrix_test, Y_train, Y_test = \
                            ds.deserialize(data_path + \
                                           'tfidf_matrix')
    except:
        Debug.print_exception_info()

print('initial matrix_train.shape', matrix_train.shape)
print('initial matrix_test.shape', matrix_test.shape)

if svd_dim > 0:

    if not serialized_svd:

        print('reducing dimension')
        matrix_train, matrix_test, svd = \
                                reduce_dimension(matrix_train,
                                                 matrix_test,
                                                 svd_dim)

        ds.serialize(svd, data_path + 'svd_' + str(svd_dim))

        ds.serialize((matrix_train, matrix_test),
                      data_path + 'lsi_matrixes_' + str(svd_dim))

    else:
        try:
            matrix_train, matrix_test = ds.deserialize(data_path + \
                                                      'lsi_matrixes_' + \
                                                      str(svd_dim))
        except:
            Debug.print_exception_info()


print(2*'\n')
print('matrix_train.shape: ', matrix_train.shape)
print('matrix_test.shape: ', matrix_test.shape)


clf1 = get_classifier(clf_name, (trees_amount,))

meta = {'ovr':OneVsRestClassifier(clf1),
        'ovo':OneVsOneClassifier(clf1),
        'occ':OutputCodeClassifier(clf1, code_size=2, random_state=0)}

#
print()
print(clf1.__class__)

clf = meta[meta_name]

print('\nfitting classifyer ...')

clf.fit(matrix_train, Y_train)

predictions = clf.predict(matrix_test)

report = get_accuracy(predictions,
                      Y_test,
                      clf,
                      data,
                      train_perc)


fname = 'report.csv.txt'

# write headers (labels separated by tab)
if not os.path.isfile(data_path + 'report.csv.txt'):
    f = open(data_path + fname, 'a')
    for label in sorted(report):
        f.write(label + '\t')
    f.write('\n')
    f.close()

f = open(data_path + fname, 'a')

for label in sorted(report):
    f.write(str(report[label]['accuracy']) + '\t')
f.write('\n')
f.close()











#############################################################

print('\n-------------------------------------------------------\n')
sec = time.time() - tt
min_ = int(sec/60)
ss = round(sec - 60*min_, 2)
print('\ntime == ', min_, ' min ', ss, ' sec')
