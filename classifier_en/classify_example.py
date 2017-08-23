# -*- coding: utf-8 -*-

import os
import time

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

from classifier import Classifier

############################################

# txt documents here
txt_data_path = '/home/yuriy/data/pyhk/txt/'

# directory for classifier files needed for work
clf_data_path = '/home/yuriy/data/pyhk/classifier/'

############################################

tt = time.time()

clf = Classifier(txt_data_path,
                 clf_data_path)

files = os.listdir(clf_data_path)
if any([not 'clf' in files,
        not 'svd' in files,
        not 'vectorizer' in files]):
    clf.relearn() # use if you want to update the model

jobText = """Accounting Supervisor We require an accounting supervisor
             for our office in Dublin. Your duties including accounting
             and answering the phone."""

category_num, category_name = clf.classify(jobText)

msg = '\npredicted category: ' + category_name
msg += ' (' + category_num + ')'
print(msg)

######################################################################

sec = time.time() - tt
min_ = int(sec/60)
ss = round(sec - 60*min_, 2)
print('\ntime == ', min_, ' min ', ss, ' sec')





