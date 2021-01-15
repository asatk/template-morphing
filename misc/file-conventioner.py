import re
import os
import time

# <KWRD>fitter-<FIT>-<VAR>-<ETA/PRIME>-<4-NUM MASS>.json
file_type = r'json'
file_suffix_len = len(file_type)+1
file_convention = r'\s*fitter-\s+-\s+-eta\s*-{{4}}\.{}'.format(file_type)
f_regex = re.compile(file_convention)
number_conv = r'(0|1)\d{{3}}\.{}'.format(file_type)
n_regex = re.compile(number_conv)
dir_name = r'./fit-plots/'
dir_list = os.listdir(dir_name)
dir_list = [s for s in dir_list if s[-file_suffix_len:] == r'.{}'.format(file_type)]

for file_str in dir_list:
    time.sleep(1)
    raw_str = r'{}'.format(dir_name+file_str)
    if not f_regex.match(raw_str):
        print("LOOKING AT NUMBERING: " + raw_str[-9:-5])
        if not n_regex.match(raw_str[-(4+file_suffix_len):-file_suffix_len]):
            n = int(re.search("\d+(?=\.json)",raw_str[-(4+file_suffix_len):-file_suffix_len]).group())
            print("NEW NUMBERING: %04i"%(n))
            new_str = re.sub("\\d+(?=\.%s)"%(file_type),"%04i"%(n),raw_str)
            print("GOING TO RENAME %s TO: %s"%(raw_str,new_str))
            os.rename(raw_str,r'{}'.format(new_str))
            print(raw_str + " - [CHANGED] DID NOT MATCH NUMBERING CONVENTION")
            continue
        print(raw_str + " - [NO CHANGE] DOES NOT MATCH FILE-NAMING CONVENTION")
