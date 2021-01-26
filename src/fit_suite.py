import os
import sys
import json

from collections import deque

import build
from build.builder import builder
# import analyze
import view
from view.viewer import viewer
# import interpolate

print sys.path
sys.path.append(os.getcwd())
print sys.path

if __name__ == "__main__":
    config_file = sys.argv[1]
    print "Using Config File: %s/%s"%(os.getcwd(),config_file)
    json_obj = None
    with open(sys.argv[1],'r') as json_file:
        json_obj = json.load(json_file)
        json_file.close()

    # BUILD
    print "[BUILDING]"
    builder_q = deque(json_obj['Qbuild'])
    file_name = json_obj['file_name']
    fit_name = json_obj['fit_name']
    fit_info = json_obj['fit_info']
    B = builder(file_name,fit_info,fitted=False,q=builder_q)
    print "Builder Command Queue:",B.q
    B.build()

    analyzer_args = json_obj['Qanalyze']

    # VIEW
    print "[VIEWING]"
    viewer_q = deque(json_obj['Qview'])
    fit_info_list = json.load(open(os.getcwd()[:os.getcwd().rfind('src')+3]+'/config.json','r'))['fits']
    vwr = viewer(fit_info_list)
    vwr.view_fits()
    vwr.list_pars()

    interpolater_args = json_obj['Qinterpolate']

    print "[END]"