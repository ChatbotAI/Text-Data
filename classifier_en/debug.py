# -*- coding: utf-8 -*-

import sys
import traceback
from pprint import pprint as pp
class Debug:
     
    @staticmethod
    def exception_info():
               
        exc_type, exc_value, exc_traceback = sys.exc_info()
        message = (traceback.format_exception_only(exc_type, exc_value))[0]
        source =  traceback.extract_tb(exc_traceback)

        return (message, source)
        
    @staticmethod
    def print_exception_info():
        pp(Debug.exception_info())
        
    

        
        
        
        