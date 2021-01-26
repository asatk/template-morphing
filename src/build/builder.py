import ROOT
import time
import os
import sys

from collections import deque

from ROOT import kAzure 
from ROOT import kPink

from fitter import fitter

class builder:
    def __init__(self,file_name,fit_info,fitted=False,q=deque()):
        self.q = q
        if fitted:
            self.ftr = fitter(fit_info)
        else:
            self.ftr = fitter(file_name,fit_info,"gaus")
        self.canv = ROOT.TCanvas("canv","title",1200,900)
        self.canv.DrawCrosshair()

    #enter fit building with cmd prompt
    def build(self):
        self.build_fit()
        cmd = " "
        while not cmd == "":
            self.canv.cd()
            if len(self.q) > 0:
                cmd = self.q.pop()
            else:
                cmd = raw_input("cmd: ")
            if cmd == 's':
                print "[SAVING]"
                self.__save()
            elif cmd == 'r' or cmd == 'refit':
                print "[REFITTING]"
                self.__refit()
            elif cmd == 'rebin':
                print "[REBINNING]"
                self.__rebin()

    #build fit with fitter
    def build_fit(self):
        self.canv.cd()        

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

        ROOT.gROOT.GetListOfSpecials().Add(hist)
        ROOT.gROOT.GetListOfSpecials().Add(func)

        self.canv.Update()
        #func.Draw("HIST SAME")

    def __rebin(self):
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

    def __refit(self):
        if len(self.q) < 3:
            file_name2 = raw_input("new data: ")
            fit_name2 = raw_input("new fit function: ")
            fit_info2 = raw_input("new fit info: ")
        else:
            file_name2 = self.q.pop()
            fit_name2 = self.q.pop()
            fit_info2 = self.q.pop()

        file_name = file_name if file_name2 == "" else file_name2
        fit_name = fit_name if fit_name2 == "" else fit_name2
        fit_info = fit_info if fit_info2 == "" else fit_info2

        ftr.file_name = file_name
        ftr.fit_info = fit_info
        ftr.fit_name = fit_name
        
        self.build_fit()

    def __save(self):
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
    bldr.build()