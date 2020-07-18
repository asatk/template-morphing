import ROOT
import time
import random
import json
import os

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
    'kRed',
    'kWhite',
    'kBlack',
    'kGray',
    'kRed',
    'kGreen',
    'kBlue',
    'kYellow',
    'kMagenta',
    'kCyan',
    'kOrange',
    'kSpring',
    'kTeal',
    'kAzure',
    'kViolet',
    'kPink']

def make_fit(file_name,fit_name,fit_info):
    canv = ROOT.TCanvas("canv","title",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    ftr = fitter(file_name,fit_name,fit_info)
    ftr.fit()
    func = ftr.func
    hist = ftr.hist
    func.SetLineColor(kRed)
    func.SetLineWidth(3)
    hist.GetXaxis().SetTitle("Mass (GeV)")
    hist.GetXaxis().CenterTitle(True)
    hist.GetYaxis().SetTitle("Events")
    hist.GetYaxis().CenterTitle(True)
    hist.SetStats(0)
    hist.SetFillColor(kAzure-8)
    hist.SetLineColor(kAzure-7)
    hist.SetLineWidth(3)
    
    #func.DrawIntegral("same")

    #print "intg func",func.Integral(ftr.lo,ftr.hi)
    #print "intg func",func.DrawIntegral("same").Eval(4.9)

    lgn = ROOT.TLegend(0.675,0.8,0.875,0.875)
    #lgn.AddEntry("hist","fit data","F")
    #lgn.AddEntry("func","%s fit"%(ftr.fit_name),"L")

    lgn.Draw("same")

    cmd = " "
    delay = 0.
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        cmd = raw_input("cmd:")
        if cmd == 's':
            print "[SAVING]"
            affirm = raw_input("are you sure you want to SAVE this model? [y/n]")
            if affirm == 'y':
                ftr.jsonify()
                #canv.SaveAs("./fit-plots/fitter-%s-%s-%s.jpg" % (ftr.fit_name,ftr.pname,
                #    ftr.file_name[ftr.file_name.find('eta'):-5]))
        elif cmd == 'r' or cmd == 'refit':
            print "[REFITTING]"
            affirm = raw_input("are you sure you want to REFIT? [y/n]")
            file_name2 = raw_input("new data: ")
            fit_name2 = raw_input("new fit function: ")
            fit_info2 = raw_input("new fit info: ")

            file_name = file_name if file_name2 == "" else file_name2
            fit_name = fit_name if fit_name2 == "" else fit_name2
            fit_info = fit_info if fit_info2 == "" else fit_info2

            if affirm == 'y':
                canv.Clear()
                ftr = fitter(file_name,fit_name,fit_info)
                ftr.fit()
                func = ftr.func
                hist = ftr.hist
                func.SetLineColor(kRed)
                func.SetLineWidth(3)
                hist.GetXaxis().SetTitle("Mass (GeV)")
                hist.GetXaxis().CenterTitle(True)
                hist.GetYaxis().SetTitle("Events")
                hist.GetYaxis().CenterTitle(True)
                hist.SetStats(0)
                hist.SetFillColor(kAzure-8)
                hist.SetLineColor(kAzure-7)
                hist.SetLineWidth(3)
                hist.Draw()
                func.Draw("c same")
                canv.Update()
        elif cmd == 'rebin':
            print "[REBINNING]"
            tempbins = raw_input("new bin size: ")
            templo = raw_input("new lo: ")
            temphi = raw_input("new hi: ")
            if tempbins is not '':
                ftr.bins = int(tempbins)
            if templo is not '':
                ftr.lo = float(templo)
            if temphi is not '':
                ftr.hi = float(temphi)
        elif cmd == '*':
            print "* - [RESTARTING]"
            affirm = raw_input("* - are you sure you want to RESTART with a new file? [y/n]")
            if affirm == 'y':
                file_name = raw_input("* - new data: ")
                fit_name = raw_input("* - fit name: ")
                ftr = fitter(file_name,fit_name=fit_name,fit_info=fit_info)
        elif cmd == 't':
            print "[DELAY]"
            delay = float(raw_input("New Delay Time (in sec):"))
        time.sleep(delay)

def get_fit(fit_info,q=deque()):
    #set up canvas
    if 'canv' in locals():
        canv.Clear()
    canv = ROOT.TCanvas("canv","title",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    #set up fitter and fit info
    json_file = open(fit_info,'r')
    info = json.load(json_file)
    file_name = info['file_name']
    ftr = fitter(file_name,fitted=True,fit_info=fit_info)
    ftr.func.SetLineColor(kPink)
    ftr.func.SetLineWidth(5)
    ftr.func.SetNpx(ftr.bins)
    func = ftr.func.Clone()

    cmd = " "
    normalized = info['normalized'] == 'True' or info['normalized'] == 'true'
    hasLegend = False
    drawfunc = False
    drawhist = False
    drawfhist = False
    delay = 0.

    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        if len(q) > 0:
            cmd = q.pop()
        else:
            print ['s','d','l','i','c','n','json-p','fhist','h','p','+','*','/','cmd-q','t']
            cmd = raw_input("cmd: ")
        if cmd == 's':
            print "s - [SAVING]"
            json_name = "./fit-files/%sfitter-%s-%s-%s.json" % ("norm" if normalized else "",ftr.fit_name,ftr.pname,
                    ftr.file_name[ftr.file_name.find('eta'):-5])
            if len(q) > 0:
                affirm = q.pop()
            else:
                affirm = raw_input("s - are you sure you want to SAVE this MODEL?\n\
                        will save as %s :[y/n]"%(json_name))
            if affirm == 'y':
                if normalized:
                    ftr.func.SetParameter(0,func.GetParameter(0))
                ftr.jsonify(fit_info=json_name)
            jpg_name = "./fit-plots/%sfitter-%s-%s-%s.jpg" % ("norm" if normalized else "",ftr.fit_name,ftr.pname,
                    ftr.file_name[ftr.file_name.find('eta'):-5])
            if len(q) > 0:
                affirm2 = q.pop()
            else:
                affirm2 = raw_input("s - are you sure you want to SAVE this IMAGE?\n\
                        will save as %s :[y/n]"%(jpg_name))
            if affirm2 == 'y':
                canv.SaveAs(jpg_name)
        elif cmd == 'd':
            print "d - [DRAWING]"
            drawhist = True
            if len(q) > 0:
                draw_opts = q.pop()
            else:
                draw_opts = raw_input("draw option input string [SAME, L, C, FC, HIST]: ")
            if "same" not in draw_opts or "SAME" not in draw_opts:
                drawhist = False
            if "hist" not in draw_opts or "HIST" not in draw_opts:
                fhist = False
            else: 
                if drawfhist:
                    del fhist
                fhist = ROOT.TH1D(func.GetHistogram())
                fhist.Rebin(fhist.GetNbinsX()/ftr.bins)
                drawfhist = True
            func.Draw(draw_opts)
            drawfunc = True
            if hasLegend:
                lgn_fit = lgn.AddEntry("func","%s fit"%(ftr.fit_name),"l")
                lgn_fit.SetLineWidth(4)
                lgn_fit.SetLineColor(func.GetLineColor())
                lgn.Draw("same")
            canv.Update()
        elif cmd == 'l':
            print "l - [LEGEND]"
            if not hasLegend:
                hasLegend = True

                #ROOT.gStyle.SetOptStat(1)

                lgn = ROOT.TLegend(0.55,0.65,0.9,0.9)
                lgn.SetHeader("Fit Function and Histogram")
                lgn.SetEntrySeparation(0.05)
                lgn.SetTextSize(0.025)
                lgn.Draw("same")
            else:
                hasLegend = False
                lgn.Clear()
                del lgn
            canv.Update()
        elif cmd == 'i':
            print "i - [INTEGRATING]"
            if len(q) > 0:
                intglo = float(q.pop())
                intghi = float(q.pop())
            else:
                intglo = float(raw_input("i - integrate from: "))
                intghi = float(raw_input("i - integrate to: "))
            print ftr.func.Integral(intglo,intghi)
            print func.Integral(ftr.lo,ftr.hi)
            canv.Update()
        elif cmd == 'c':
            print "c - [CUMULATIVE]"
            #fintg = func.DrawIntegral(option="")
            if drawfhist:
                fhist.GetCumulative().Draw()
                fhist.GetIntegral()
                print "c - fhist getintegral",fhist.Integral()
            #func.DrawDerivative("same")
            canv.Update()
        elif cmd == 'n':
            print "n - [NORMALIZING]"
            if not normalized:
                if drawfhist:
                    del fhist
                
                fhist = ROOT.TH1D(func.GetHistogram())
                norm_factor = 1. / fhist.Integral()
                func.SetParameter(0,func.GetParameter(0)*norm_factor)

                if drawhist:
                    hist = ROOT.TH1D(hist.DrawNormalized())
            
                if drawfhist:
                    if 'fhist_norm' in locals():
                        del fhist_norm
                    fhist_norm = ROOT.TH1D(fhist.DrawNormalized("HIST"+("SAME" if drawhist else "")))
                    fhist_norm.SetTitle("fhist_norm")
                    fhist_norm.SetName("fhist_norm")
                    canv.Update()
                
                if not drawfhist and drawhist:
                    hist.Draw("HIST")
                    canv.Update()
                if drawfunc:
                    func.SetLineColor(kPink)
                    func.Draw("C"+("SAME" if fhist else ""))
                    canv.Update()
                canv.Update()
                normalized = True
            elif normalized:
                print "n - de-normalizing"
                norm_factor = 1. if 'norm_factor' not in locals() else 1. / norm_factor
                func.SetParameter(0,func.GetParameter(0)*norm_factor)
                func.SetLineColor(kPink)
                if drawhist:
                    hist.Scale(norm_factor)
                    hist.Draw("HIST")
                    canv.Update()
                if drawfunc:
                    func.Draw("C"+("SAME" if drawhist else ""))
                    canv.Update()
                normalized = False
            
            if hasLegend:
                lgn.Draw("same")

            print "n - normalization factor:",norm_factor
            canv.Update()
        elif cmd == 'json-p':
            print "json-p - [PRINTING STATS]"
            with open(fit_info) as json_file:
                info = json.load(json_file)
                print json.dumps(info,indent=4)
        elif cmd == 'fhist':
            if drawfhist:
                del fhist
                drawfhist = False
            else:
                if 'fhist' in locals():
                    del fhist
                fhist = ROOT.TH1D(func.GetHistogram())
                fhist.Rebin(fhist.GetNbinsX()/ftr.bins)
                fhist.Draw("same")
                drawfhist = True
            canv.Update()
        elif cmd == 'h':
            print "h - [HISTOGRAM PLOTTING]"
            if drawhist:
                del hist
                del chain
            hist = ROOT.TH1D("hist","%s fit - %s"%(ftr.fit_name,ftr.var),ftr.bins,ftr.lo,ftr.hi)
            hist.GetXaxis().SetTitle("Mass (GeV)")
            hist.GetXaxis().CenterTitle(True)
            if not normalized:
                hist.GetYaxis().SetTitle("Events")
            hist.GetYaxis().CenterTitle(True)
            hist.SetStats(0)
            hist.SetFillColor(kAzure-8)
            hist.SetLineColor(kAzure-7)
            hist.SetLineWidth(3)
            chain = ROOT.TChain("twoprongNtuplizer/fTree")
            chain.Add(info['file_name'])
            draw_s = info['var'] + ">>hist"
            cut_s = info['cuts']
            chain.Draw(draw_s, cut_s)
            if normalized:
                hist = ROOT.TH1D(hist.DrawNormalized())

            fhist = func.GetHistogram()
            fhist.Rebin(fhist.GetNbinsX()/ftr.bins)
            if drawfhist:
                func.Draw("same HIST")

            if hasLegend:
                lgn.AddEntry("hist",ftr.file_name[ftr.file_name.rfind('/')+1:],"f")
                lgn.Draw("same")

            drawhist = True
            canv.Update()

            print "h - hist intg",hist.Integral()
            print "h - hist intg width",hist.Integral("width")
            print "h - fhist intg",fhist.Integral()
            print "h - fhist intg width",fhist.Integral("width")
        elif cmd == 'p':
            print "p - [FETCHING PARAMETERS]"
            par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
            par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
            pars = zip(par_names,par_values)
            print "p - ",pars
            print "p - Chi-Squared",ftr.chi
        elif cmd == '+':
            print "+ - [ADDING FILE, NORMALIZING]"

            if "fit_info2" in locals():
                fit_info = fit_info2
                ftr = ftr2
                func = func2

            file_name2 = raw_input("+ - new fit data to add: ")
            fit_info2 = raw_input("+ - new fit init file to add: ")
            ftr2 = fitter(file_name2,fitted=True,fit_info=fit_info2)
            ftr2.func.SetNpx(ftr2.bins)
            func2 = ftr2.func.Clone()
            func2.SetLineColor(kBlue-3)
            func2.SetLineWidth(5)
            
            func.Draw()
            func2.Draw("same")

            print "+ - 1st func X^2",ftr.chi
            print "+ - 2nd func X^2",ftr2.chi

            par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
            par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
            par_values2 = list(func2.GetParameter(i) for i in range(func.GetNpar()))
            par_avgs = list((par_values2[i] + par_values[i])/2 for i in range(func.GetNpar()))
            par_diffs = list(par_values2[i] - par_values[i] for i in range(func.GetNpar()))
            par_pcts = list(abs(par_diffs[i])/par_avgs[i] for i in range(func.GetNpar()))
            pars = zip(par_names,par_avgs,par_pcts)
            
            print "+ - parameter comparison\n",pars
            canv.Update()
        elif cmd == "*":
            print "* - [RESTARTING]"
            if len(q) > 0:
                affirm = q.pop()
            else:
                affirm = raw_input("* - are you sure you want to RESTART with a new file? [y/n]")
            if affirm == 'y':
                if 'lgn' in locals():
                    lgn.Clear()
                canv.Clear()
                
                fit_info = raw_input("* - name of fit info .json: ")

                json_file = open(fit_info,'r')
                info = json.load(json_file)
                file_name = info['file_name']
                ftr = fitter(file_name,fitted=True,fit_info=fit_info)
                ftr.func.SetNpx(ftr.bins)
                func.SetLineColor(kPink)
                func.SetLineWidth(5)
                func = ftr.func.Clone()

                normalized = info['normalized'] == 'True' or info['normalized'] == 'true'
                hasLegend = False
                drawfunc = False
                drawhist = False
                drawfhist = False
                delay = 0.
                q = deque()
                canv.Update()
        elif cmd == '/':
            print "/ - [CLEARING]"
            func = ftr.func.Clone()
            func.SetLineColor(kPink)
            func.SetLineWidth(5)
            if 'lgn' in locals():
                lgn.Clear()
            canv.Clear()
            
            delay = 0.
            normalized = info['normalized'] == 'True' or info['normalized'] == 'true'
            hasLegend = False
            drawfunc = False
            drawhist = False
            drawfhist = False
            q = deque()
            canv.Update()
        elif cmd == 'cmd-q':
            print "cmd-q - [COMMAND QUEUE]"
            qstring = raw_input("cmd-q - input string of commands separated by ',' (1 line): ")
            qlist = qstring.split(',')
            qlist.reverse()
            q = deque(qlist)
            print "cmd-q - queue:",q
        elif cmd == 't':
            print "t - [DELAY]"
            delay = float(raw_input("t - New Delay Time (in sec):"))
        time.sleep(delay)

if __name__ == '__main__':
    file_name = 'root/TwoProngNtuplizer_eta125.root'
    fit_name = 'crystalball'
    
    cmd = " "
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        cmd = raw_input("run:")
        if cmd in ['m','make','make fit','make_fit']:
            fit_info = 'fitter-init.json'
            make_fit(file_name,fit_name,fit_info)
        elif cmd in ['g','get','get fit','get_fit']:
            fit_info = './fit-files/fitter-%s-omega-eta750.json'%(fit_name)
            get_fit(fit_info)
        elif cmd in ['n','na','norm','norm all','norm-all']:
            fit_dir = "./fit-files/"
            for f in os.listdir("./fit-files"):
                if f[:4] != "norm":
                    q = deque(['','c same','d','h','n','l'])
                    print "normalizing fit for %s"%(f)
                    get_fit("./fit-files/"+f,q)
                    affirm = raw_input("fit display and data obtained for %s: next file? [y/n]"%(f))
                    if affirm != 'y':
                        break