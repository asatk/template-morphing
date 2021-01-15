import ROOT
import time
import random
import json
import os
import sys
import math
import re

from collections import deque

from ROOT import Math
from ROOT import TMath

from ROOT import kRed
from ROOT import kWhite
from ROOT import kBlack
from ROOT import kGray
from ROOT import kRed
from ROOT import kGreen 
from ROOT import kBlue 
from ROOT import kYellow 
from ROOT import kMagenta 
from ROOT import kCyan 
from ROOT import kOrange 
from ROOT import kSpring 
from ROOT import kTeal 
from ROOT import kAzure 
from ROOT import kViolet 
from ROOT import kPink

from fitter import fitter

ROOTCOLORS = [
    kPink+1,
    kOrange+1,
    kYellow,
    kGreen-9,
    kRed,
    kGreen,
    kBlue,
    kCyan,
    kMagenta,
    kYellow,
    kGray,
    kBlack,
    kOrange,
    kSpring,
    kTeal,
    kAzure,
    kViolet,
    kPink]

ROOTCOLORS2 = [
    kRed,
    kGreen,
    kBlue-3,
    kViolet,
    kOrange
]

class builder:
    def __init__(self,file_name,fit_name,fit_info,fitted=False,q=deque()):
        self.q = q
        self.ftr = fitter(file_name,fit_name,fit_info,fitted=fitted)

    def build_fit(self):
        canv = ROOT.TCanvas("canv","title",1200,900)
        canv.DrawCrosshair()
        canv.cd()

        self.ftr.fit()
        self.ftr.func.SetLineColor(kPink)
        self.ftr.func.SetLineWidth(5)
        self.ftr.func.SetNpx(self.ftr.bins)
        func = self.ftr.func.Clone()
        hist = self.ftr.hist.Clone()
        hist.GetXaxis().SetTitle("%s (GeV)"%(self.ftr.var))
        hist.GetXaxis().CenterTitle(True)
        hist.GetYaxis().SetTitle("Events")
        hist.GetYaxis().CenterTitle(True)
        hist.SetStats(1)
        ROOT.gStyle.SetOptFit(1111)
        ROOT.gStyle.SetOptStat(1111)
        hist.SetFillColor(kAzure-8)
        hist.SetLineColor(kAzure-7)
        hist.SetLineWidth(3)

        hist.Draw("HIST")
        func.Draw("C SAME")
        #func.Draw("HIST SAME")

        cmd = " "
        while not cmd == "" and\
            not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
            if len(self.q) > 0:
                cmd = self.q.pop()
            else:
                cmd = raw_input("cmd: ")
            if cmd == 's':
                print "[SAVING]"
                self.save()
            elif cmd == 'r' or cmd == 'refit':
                print "[REFITTING]"
                self.__refit__()
            elif cmd == 'rebin':
                print "[REBINNING]"
                self.__rebin__()

    def __rebin__(self):
        if len(self.q) < 3:
            tempbins = raw_input("new bin size: ")
            templo = raw_input("new lo: ")
            temphi = raw_input("new hi: ")
        else:
            tempbins = self.q.pop()
            templo = self.q.pop()
            temphi = self.q.pop()
        if tempbins is not '':
            self.ftr.bins = int(tempbins)
        if templo is not '':
            self.ftr.lo = float(templo)
        if temphi is not '':
            self.ftr.hi = float(temphi)

    def __refit__(self):
        if len(self.q) > 0:
            affirm = self.q.pop()
        else:
            affirm = raw_input("are you sure you want to REFIT? [y/n]")
        if affirm == 'y':
            if len(self.q) < 3:
                file_name2 = raw_input("new data: ")
                fit_name2 = raw_input("new fit function: ")
                fit_info2 = raw_input("new fit info: ")
            else:
                file_name2 = self.q.pop()
                fit_name2 = self.q.pop()
                fit_info = self.q.pop()

        file_name = file_name if file_name2 == "" else file_name2
        fit_name = fit_name if fit_name2 == "" else fit_name2
        fit_info = fit_info if fit_info2 == "" else fit_info2

        if affirm == 'y':
            canv.Clear()
            self.ftr = fitter(file_name,fit_name,fit_info)
            self.ftr.fit()
            self.ftr.func.SetLineColor(kPink)
            self.ftr.func.SetLineWidth(5)
            self.ftr.func.SetNpx(self.ftr.bins)
            func = self.ftr.func.Clone()
            hist = self.ftr.hist.Clone()
            hist.GetXaxis().SetTitle("%s (GeV)"%(self.ftr.var))
            hist.GetXaxis().CenterTitle(True)
            hist.GetYaxis().SetTitle("Events")
            hist.GetYaxis().CenterTitle(True)
            hist.SetStats(1)
            hist.SetFillColor(kAzure-8)
            hist.SetLineColor(kAzure-7)
            hist.SetLineWidth(3)
            hist.Draw()
            func.Draw("c same")
            #func.Draw("hist same")
            canv.Update()

    def __save__(self):
        if len(self.q) > 0:
            affirm = self.q.pop()
        else:
            affirm = raw_input("are you sure you want to SAVE this model? [y/n]")
        if affirm == 'y':
            self.ftr.jsonify()

if __name__ == "__main__":
    print sys.argv
    print sys.path
    bldr = builder(sys.argv[1],sys.argv[2],sys.argv[3],bool(sys.argv[4]),deque())
    bldr.build_fit()