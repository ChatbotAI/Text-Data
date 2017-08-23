# -*- coding: utf-8 -*-

import pickle

class DataSerializer:
    """ objects to files serialization"""
    
    @staticmethod
    def serialize(obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
            
    
    @staticmethod
    def deserialize(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)