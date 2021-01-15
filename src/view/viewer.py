import ROOT
import time
import random
import json
import os
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

from build.fitter import fitter

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

def view_fits(fit_info_list,num_mass_pts=2,saveViews=False,opt="HIST",q=deque()):
    
    interp_flist = []

    #set up canvas
    if 'canv' in locals():
        canv.Clear()
    canv = ROOT.TCanvas("canv","plot analysis",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    with open(fit_info_list[0],'r') as json_file:
        info = json.load(json_file)
        file_name = info['file_name']
        ftr = fitter(file_name,fitted=True,fit_info=fit_info_list[0])
    
    masspts = []
    for f in fit_info_list:
        if ftr.pname == 'phi':
            masspt = float(re.search("(\d+)(?=\.json)", f).group())
        elif ftr.pname == 'omega':
            masspt = 0.95 if bool(re.search("prime", f) is not None) else 0.55
        if masspt not in masspts:
            masspts.append(masspt)
            
    normalized = 'norm' in fit_info_list[0]
    # num_mass_pts = 2
    pdf_color = kBlack
    Npx = ftr.bins
    cmd = " "

    print "pdf - [PDFs]"
    hstack = ROOT.THStack("hs","%s of %s for %s"%("Probability Distributions" if normalized else "Event Distributions",ftr.var,ftr.pname))
    flist = ROOT.TList()

    lgn = ROOT.TLegend(0.55,0.55,0.89,0.88)
    lgn.SetTextFont(22)
    lgn.SetHeader("Fit Function Eta/Eta' Chi^2")
    lgn.SetTextFont(132)
    lgn.SetEntrySeparation(0.05)
    lgn.SetTextSize(0.025)
    lgn.SetLineWidth(2)
    lgn.SetFillColor(19)

    for count,i in enumerate(fit_info_list):
        # set up fitter and fit info
        print "using fit model #%i: %s"%(count,i)
        json_file = open(i,'r')
        info = json.load(json_file)
        file_name = info['file_name']

        ftr = fitter(file_name,fitted=True,fit_info=i)
        if num_mass_pts == 5:
            ftr.func.SetLineColor((ROOTCOLORS2[count%5] + (-4 if count > 4 else 2)))
        elif num_mass_pts == 2:
            ftr.func.SetLineColor(ROOTCOLORS2[count//5 + 1] + 1 - 2*(count%5))
        else:
            ftr.func.SetLineColor(pdf_color)
        ftr.func.SetLineWidth(4)
        ftr.func.SetNpx(Npx)
        func = ftr.func.Clone()
        interp_flist.append(func)
        fhist = ROOT.TH1D(func.GetHistogram())
        hstack.Add(fhist)

        eta_idx = i.find('eta')
        ext_idx = i[eta_idx:].find('.')

        lgn_func = lgn.AddEntry(func,"%s - %4.4f"%(i[eta_idx:eta_idx + ext_idx],ftr.chi),"l")
        lgn_func.SetLineWidth(4)
        lgn_func.SetLineColor(func.GetLineColor())

        json_file.close()

    hasDrawn = False
    if opt == "HIST":
        print hstack.GetNhists()
        # hstack.GetHists().Print()
        hstack.Draw("HIST nostack")
        canv.Update()
        hasDrawn = True
    elif opt == "FUNC":
        hasDrawn = False
    if opt == "BOTH" or opt == "FUNC":
        for interp_f in interp_flist:
            y_ax = interp_f.GetYaxis()
            y_ax.SetRangeUser(0,1.0)
            interp_f.Draw("C" + "SAME" if hasDrawn else "")
            canv.Update()
            hasDrawn = True

    lgn.Draw()
    canv.Update()
    
    # print("[' ', 's']")
    # cmd = raw_input("cmd: ")
    # if cmd == 's':
    if saveViews:
        eta_idx = fit_info_list[0].find('eta')
        p = r'^(\.\/.+)\/(\w*fitter)'
        fitter_idx = re.search(p,fit_info_list[0]).start(2)
        canv.SaveAs('./fit-plots/' + fit_info_list[0][fitter_idx:eta_idx-1] + '-' +  opt.lower() +  '.jpg')

def list_pars(fit_info_list):

    with open(fit_info_list[0],'r') as json_file:
        info = json.load(json_file)
        file_name = info['file_name']
        ftr = fitter(file_name,fitted=True,fit_info=fit_info_list[0])

    for count,i in enumerate(fit_info_list):
        # set up fitter and fit info
        # print "using fit model #%i: %s"%(count,i)
        json_file = open(i,'r')
        info = json.load(json_file)
        file_name = info['file_name']

        ftr = fitter(file_name,fitted=True,fit_info=i)
        func = ftr.func.Clone()

    # print "p - [FETCHING PARAMETERS]"
        par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
        par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
        pars = zip(par_names,par_values)
        print str(par_values)[1:-1]
        # print par_names
        # print "p - ",pars
        # print "p - Chi-Squared",ftr.chi

if __name__ == "__main__":
    print sys.argv
    # view_fits()
    # list_fits()