import os
import sys
import json
import subprocess

import build
from build.builder import builder
import analyze
import view
import interpolate

if __name__ == "__main__":
    config_file = sys.argv[1]
    print "Using Config File: %s/%s"%(os.getcwd(),config_file)
    json_obj = None
    with open(sys.argv[1],'r') as json_file:
        json_obj = json.load(json_file)
        json_file.close()

    # BUILD
    print "[BUILDING]"
    builder_q = json_obj['Qbuild']
    file_name = json_obj['file_name']
    fit_name = json_obj['fit_name']
    fit_info = json_obj['fit_info']
    B = builder(file_name,fit_name,fit_info,q=builder_q)
    print "Builder Command Queue:",B.q
    B.build_fit()

    analyzer_args = json_obj['Qanalyze']
    viewer_args = json_obj['Qview']
    interpolater_args = json_obj['Qinterpolate']

    print "[END]"