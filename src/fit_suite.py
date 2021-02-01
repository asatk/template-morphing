import os
import sys
import json

from collections import deque

import build
from build.builder import builder
import analyze
from analyze.analyzer import analyzer
import view
from view.viewer import viewer
# import interpolate

# print sys.path
sys.path.append(os.getcwd())
# print sys.path

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
    B = builder(q=builder_q)
    print "Builder Command Queue:",B.q
    B.build()

    # ANALYZE
    # analyzer_q = deque(json_obj['Qanalyze'])
    # fit_info_list = json_obj['fit_info_list']
    # fit_info0 = fit_info_list[0]
    # A = analyzer(fit_info0,q=analyzer_q)
    # print "Analyzer Command Queue:",A.q
    # A.analyze()
    # print "Analyzer Fit Info List:",A.ftr.

    # VIEW
    # print "[VIEWING]"
    # viewer_q = deque(json_obj['Qview'])
    # fit_info_list = json.load(open(os.getcwd()[:os.getcwd().rfind('src')+3]+'/config.json','r'))['fit_info_list']
    # vwr = viewer(fit_info_list,q=viewer_q)
    # print "Viewer Fit Info List:",vwr.fit_info_list
    # vwr.view_fits()
    # vwr.list_pars()

    # INTERPOLATE
    # interpolater_q = deque(json_obj['Qinterpolate'])

    print "[END]"