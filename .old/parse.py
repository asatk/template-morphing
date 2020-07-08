import sys
import os
import json
import re

def parse (fileName="output.txt"):
    print "File: ",fileName
    with open(fileName) as file:
        line = file.readline()
        samples = []
        variable = ""
        cuts = ""

        newdata = True

        if not line.find("It appears")+1:
            print("Please use the --verb option in plotter.py call")
            exit
        collectvars = False
        while line:
            print "\nLine: ",line[:len(line)-1]
            collectfiles = bool(line.find("It appears")+1)
            print "cFiles: ",collectfiles
            if collectfiles:
                samples.append(
                    (line)[line.find("eta"):line.find(".root",1)])
                print "samples: ",samples
            
            collectvars = bool(line.find("Draw String")+1)
            print "cVars: ",collectvars
            if collectvars:
                varidx = len("Draw String: ")+1
                variable = line[varidx:line.find(">>")]
                cutsidx = line[varidx:].find(' ')+varidx+1
                cuts = line[cutsidx:]
                print "variable: ",variable,"\tcuts: ",cuts
            
            line = file.readline()
        



if __name__ == '__main__':
    #fileName = raw_input("filename?: ")
    #parse(fileName)
    parse()
    print("JSON Parsing Complete")