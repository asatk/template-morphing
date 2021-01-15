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

def analyze_fit(fit_info,q=deque()):
    # set up canvas
    if 'canv' in locals():
        canv.Clear()
    canv = ROOT.TCanvas("canv","title",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    # set up fitter and fit info
    json_file = open(fit_info,'r')
    info = json.load(json_file)
    file_name = info['file_name']
    print "USING MODEL",fit_info
    print "USING DATA FROM",file_name
    ftr = fitter(file_name,fitted=True,fit_info=fit_info)
    ftr.func.SetLineColor(kPink)
    ftr.func.SetLineWidth(5)
    # ftr.func.SetNpx(ftr.bins*5)
    ftr.func.SetNpx(ftr.bins)
    func = ftr.func.Clone()

    cmd = " "
    normalized = info['normalized'] == 'True' or info['normalized'] == 'true'
    hasLegend = False
    hasStats = False
    drawfunc = False
    drawhist = False
    drawfhist = True
    drawcum = False

    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        if len(q) > 0:
            cmd = q.pop()
        else:
            print ['s','d','l','lstats','i','c','n','h','line','p','json-p','+','*','/','cmd-q']
            cmd = raw_input("cmd: ")
        if cmd == 's':
            print "s - [SAVING]"
            if normalized: keyword = "norm"
            elif drawcum: keyword = "cum"
            else: keyword = ""
            eta_start = file_name.find('eta')
            num_match = re.search("\\d+(?=\\.root)",ftr.file_name[eta_start:])
            num = int(num_match.group())
            num_start = int(num_match.start())
            padded_num_name = re.sub("\\d+(?=\\.json)","%04i"%(num),ftr.file_name)
            json_name = "./fit-files/%sfitter-%s-%s-%s.json" % (keyword,ftr.fit_name,ftr.pname,
                    ftr.file_name[eta_start:eta_start + num_start] + "%04i"%(num))
            if len(q) > 0:
                affirm1 = q.pop()
            else:
                affirm1 = raw_input("s - are you sure you want to SAVE this MODEL?\n\
                        will save as %s :[y/n]"%(json_name))
            if affirm1 == 'y':
                if normalized:
                    ftr.func.SetParameter(0,func.GetParameter(0))
                ftr.jsonify(fit_info=json_name)
            jpg_name = "./fit-plots/%sfitter-%s-%s-%s.jpg" % (keyword,ftr.fit_name,ftr.pname,
                    ftr.file_name[eta_start:eta_start + num_start] + "%04i"%(num))
            if len(q) > 0:
                affirm2 = q.pop()
            else:
                affirm2 = raw_input("s - are you sure you want to SAVE this IMAGE?\n\
                        will save as %s :[y/n]"%(jpg_name))
            if affirm2 == 'y':
                canv.SaveAs(jpg_name)
        elif cmd == 'd':
            print "d - [DRAWING]"
            if len(q) > 0:
                draw_opts = q.pop()
            else:
                draw_opts = raw_input("draw option input string [SAME, L, C, FC, HIST]: ")
            
            if not("same" in draw_opts or "SAME" in draw_opts):
                drawcum = False
                canv.Clear()
            if drawhist:
                hist.Draw("HIST")
                canv.Update()

            if len(draw_opts) == 0\
                or "l" in draw_opts or "L" in draw_opts\
                or "c" in draw_opts or "C" in draw_opts\
                or "fc" in draw_opts or "FC" in draw_opts:
                drawfunc = True
            else:
                drawfunc = False
                if "hist" in draw_opts or "HIST" in draw_opts:
                    drawfhist = True

            func.Draw(draw_opts + ("SAME" if drawhist else ""))
            canv.Update()
            if hasLegend:
                lgn = legend(lgn,"FUNC",func=func,fit_name=ftr.fit_name)
                lgn.Draw()
                canv.Update()
            if hasStats:
                st = stats(st,"FUNC",func=func,chi=ftr.chi,NDF=ftr.NDF)
                st.Draw()
                canv.Update()
        elif cmd == 'l':
            print "l - [LEGEND]"
            if not hasLegend:
                hasLegend = True
                lgn = legend("ON")
                lgn.Draw()
                canv.Update()
                if drawfunc or drawfhist:
                    lgn = legend(lgn,"FUNC",func=func,fit_name=ftr.fit_name)
                    lgn.Draw()
                    canv.Update()
                if drawhist:
                    lgn = legend(lgn,"HIST",hist=hist,file_name=ftr.file_name)
                    lgn.Draw()
                    canv.Update()
            else:
                hasLegend = False
                legend(lgn,"OFF")
                canv.Update()
        elif cmd == 'lstats':
            print "lstats - [STATS BOX]"
            if not hasStats:
                hasStats = True
                st = stats("ON")
                st.Draw()
                canv.Update()
                if drawfunc or drawfhist:
                    st = stats(st,"FUNC",chi=ftr.chi,func=func,NDF=ftr.NDF)
                    st.Draw()
                    canv.Update()
            else:
                hasStats = False
                stats(st,"OFF")
                canv.Update()
        elif cmd == 'i':
            print "i - [INTEGRATING]"
            if len(q) > 0:
                intglo = float(q.pop())
                intghi = float(q.pop())
            else:
                intglo = float(raw_input("i - integrate from: "))
                intghi = float(raw_input("i - integrate to: "))
            print "i - integral of func on [%4.4f,%4.4f]: %f"%(intglo,intghi,func.Integral(intglo,intghi))
            print "i - integral of func on range [%4.4f,%4.4f]: %f"%(ftr.lo,ftr.hi,func.Integral(ftr.lo,ftr.hi))
            # print "i - integral of func on range [%4.4f,%4.4f]: %f"%(ftr.lo,ftr.hi,func.GetHistogram().Integral())
            canv.Update()
        elif cmd == 'c':
            print "c - [CUMULATIVE]"
            canv.Clear()
            if drawhist:
                hist.SetMaximum(1.05*max(hist.Integral(),1. if not (drawfunc or drawfhist) else func.GetHistogram().Integral()))
                hist.GetCumulative().Draw()
                canv.Update()
                print "c - hist Integral",hist.Integral()
            if drawfhist:
                func.GetHistogram().GetCumulative().Draw("SAME" if drawhist else "")
                canv.Update()
                print "c - func histogram Integral",func.GetHistogram().Integral()
            if drawfunc:
                cumfunc = func.Clone()
                cumfunc.SetParameter(0,cumfunc.GetParameter(0)/cumfunc.GetXaxis().GetBinWidth(0))
                cum = ROOT.TGraph(cumfunc.DrawIntegral("SAME" if drawhist or drawfhist else ""))
                canv.Update()
                print "c - func Integral",cumfunc.Integral(ftr.lo,ftr.hi)
                cumfunc.Print()
            drawcum = True
        elif cmd == 'n':
            if not normalized:
                print "n - [NORMALIZING]"
                hscale_factor = 1. if 'hist' not in locals() else (1. / hist.Integral())
                # func.SetNpx(hist.GetNbinsX())
                norm_factor = 1. if 'func' not in locals() else (1. / func.GetHistogram().Integral())
                if 'func' in locals():
                    func.SetParameter(0,func.GetParameter(0)*norm_factor)
                if drawhist:
                    hist.GetYaxis().SetTitle("Event Probability Density")
                    hist = ROOT.TH1D(hist.DrawNormalized("HIST"))
                    canv.Update()
                    if drawfunc or drawfhist:
                        y_ax_max = float(1.05*max([func.GetMaximum(),hist.GetMaximum()]))
                        hist.SetMaximum(y_ax_max)
                        func.SetMaximum(y_ax_max)
                        canv.Update()
                        # hist.Draw("HIST")
                if drawfhist:
                    func.GetHistogram().DrawNormalized("HIST"+("SAME" if drawhist else ""))
                    canv.Update()
                if drawfunc:
                    func.DrawCopy("C"+("SAME" if drawfhist or drawhist else ""))
                    canv.Update()
                normalized = True
            else:
                print "n - [DE-NORMALIZING]"
                hscale_factor = 1. if 'hscale_factor' not in locals() else 1. / hscale_factor
                norm_factor = 1. if 'norm_factor' not in locals() else 1. / norm_factor
                func.SetParameter(0,func.GetParameter(0)*norm_factor)
                if drawhist:
                    hist.GetYaxis().SetTitle("Events")
                    hist.Scale(hscale_factor)
                    if drawfunc or drawfhist:
                        hist.SetMaximum(float(1.05*max([func.GetMaximum(),hist.GetMaximum()])))
                    hist.Draw("HIST")
                    canv.Update()
                if drawfhist:
                    func.GetHistogram().Draw("HIST"+("SAME" if drawhist else ""))
                    canv.Update()
                if drawfunc:
                    func.Draw("C"+("SAME" if drawhist or drawfhist else ""))
                    canv.Update()
                normalized = False
            
            if hasLegend:
                if drawhist:
                    lgn = legend(lgn,"HIST",hist=hist,file_name=ftr.file_name)
                    lgn.Draw()
                    canv.Update()
            if hasStats:
                if drawfunc:
                    st = stats(st,option="FUNC",func=func,chi=ftr.chi,NDF=ftr.NDF)
                    st.Draw()
                    canv.Update()
            
            for n in range(func.GetNpar()):
                print func.GetParName(n),func.GetParameter(n)
            func.Print()
            print func
            print "n - function normalization factor:",norm_factor
            print "n - histogram scale factor:",hscale_factor
            canv.Update()
        elif cmd == 'h':
            print "h - [HISTOGRAM PLOTTING]"
            if drawhist:
                hist.Delete()
                chain.Delete()
            drawhist = True
            hist = ROOT.TH1D("hist","%s fit - %s"%(ftr.fit_name,ftr.var),ftr.bins,ftr.lo,ftr.hi)

            x_ax = hist.GetXaxis()
            if ftr.pname == 'phi':
                x_ax.SetTitle("%s (MeV)"%(ftr.var))
            else:
                x_ax.SetTitle("%s (GeV)"%(ftr.var))
            x_ax.CenterTitle(True)
            y_ax = hist.GetYaxis()
            y_ax.SetTitle("Event Probability Density" if normalized else "Events")
            y_ax.CenterTitle(True)

            hist.SetFillColor(kAzure-8)
            hist.SetLineColor(kAzure-7)
            hist.SetLineWidth(3)
            hist.SetStats(0)

            chain = ROOT.TChain("twoprongNtuplizer/fTree")
            chain.Add(info['file_name'])
            draw_s = info['var'] + ">>hist"
            cut_s = info['cuts']
            chain.Draw(draw_s, cut_s, option="HIST")
            #y_ax.SetLimits(0.,max(func.GetMaximum(),hist.GetMaximum())*1.05)
            #y_ax.SetMax(500)
            if drawfunc or drawfhist:
                hist.SetMaximum(float(1.05*max([func.GetMaximum(),hist.GetMaximum()])))
                pass
            canv.Update()

            if drawfhist:
                func.Draw("HIST SAME")
                canv.Update()
            if drawfunc:
                func.Draw("C SAME")
                canv.Update()
            if hasLegend:
                lgn = legend(lgn,"HIST",hist=hist,file_name=ftr.file_name)
                lgn.Draw()
                canv.Update()
            if hasStats:
                st.Draw()
                canv.Update()

            print "h - hist intg",hist.Integral()
            print "h - fhist intg",func.GetHistogram().Integral()
        elif cmd == 'line':
            print "line - [LINE AT MEAN]"
            par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
            par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
            pars = zip(par_names,par_values)
            print pars
            if 'Mean' in par_names:
                mean = float(pars[par_names.index('Mean')][1])
            elif 'MPV' in par_names:
                mean = float(pars[par_names.index('MPV')][1])
            else:
                mean = 0.
            val = func.Eval(mean)
            cm = func.CentralMoment(2,ftr.lo,ftr.hi)

            vline = ROOT.TLine(mean,0.,mean,func.GetMaximum())
            vline.SetLineColor(kGreen)
            vline.SetLineWidth(4)
            vline.Draw("SAME")
            hline1 = ROOT.TLine(ftr.lo,val,mean,val)
            hline1.SetLineColor(kMagenta)
            hline1.SetLineWidth(4)
            hline1.Draw("SAME")
            hline2 = ROOT.TLine(cm,0.,cm,func.GetMaximum())
            hline2.SetLineColor(kCyan)
            hline2.SetLineWidth(4)
            hline2.Draw("SAME")
            canv.Update()
            
            print "line - mean:",mean
            print "line - value at mean:",val
            print "line - central moment",cm
        elif cmd == 'p':
            print "p - [FETCHING PARAMETERS]"
            par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
            par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
            pars = zip(par_names,par_values)
            print "p - ",pars
            print "p - Chi-Squared",ftr.chi
        elif cmd == 'json-p':
            print "json-p - [PRINTING STATS]"
            print "json-p - File:",fit_info
            with open(fit_info) as json_file:
                info = json.load(json_file)
                print json.dumps(info,indent=4)
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
                print "USING DATA FROM",file_name
                ftr = fitter(file_name,fitted=True,fit_info=fit_info)
                ftr.func.SetNpx(ftr.bins)
                func = ftr.func.Clone()
                func.SetLineColor(kPink)
                func.SetLineWidth(5)

                normalized = info['normalized'] == 'True' or info['normalized'] == 'true'
                hasLegend = False
                hasStats = False
                drawfunc = False
                drawhist = False
                drawfhist = False
                drawcum = False
                q = deque()
                canv.Update()
        elif cmd == '/':
            print "/ - [CLEARING]"
            func = ftr.func.Clone()
            func.SetLineColor(kPink)
            func.SetLineWidth(5)
            if 'lgn' in locals():
                lgn.Delete()
            canv.Clear()
            
            normalized = info['normalized'] == 'True' or info['normalized'] == 'true'
            hasLegend = False
            hasStats = False
            drawfunc = False
            drawhist = False
            drawfhist = False
            drawcum = False
            q = deque()
            canv.Update()
        elif cmd == 'cmd-q':
            print "cmd-q - [COMMAND QUEUE]"
            qstring = raw_input("cmd-q - input string of commands separated by ',' (1 line): ")
            qlist = qstring.split(',')
            qlist.reverse()
            q = deque(qlist)
            print "cmd-q - queue:",q

def legend(lgn=None,option="ON",**kwargs):
    if option == "ON":
        lgn = ROOT.TLegend(0.55,0.75,0.89,0.88)
        lgn.SetTextFont(22)
        lgn.SetHeader("Fit Function and Histogram")
        lgn.SetTextFont(132)
        lgn.SetEntrySeparation(0.05)
        lgn.SetTextSize(0.025)
        lgn.SetLineWidth(2)
        lgn.SetFillColor(19)
        return lgn
    elif option == "FUNC":
        func = kwargs['func']
        lgn_entry_list = lgn.GetListOfPrimitives()
        entry_printed = False
        for obj in lgn_entry_list:
            entry = ROOT.TLegendEntry(obj)
            if entry.GetLabel() == "%s fit curve"%(kwargs['fit_name']):
                entry_printed = True
        if not entry_printed:
            lgn_func = lgn.AddEntry(func,"%s fit curve"%(kwargs['fit_name']),"l")
            lgn_func.SetLineWidth(4)
            lgn_func.SetLineColor(func.GetLineColor())
        return lgn
    elif option == "HIST":
        hist = kwargs['hist']
        file_name = kwargs['file_name']
        lgn_entry_list = lgn.GetListOfPrimitives()
        entry_printed = False
        for obj in lgn_entry_list:
            entry = ROOT.TLegendEntry(obj)
            if entry.GetLabel() == file_name[file_name.rfind('/')+1:]:
                entry_printed = True
        if not entry_printed:
            lgn_hist = lgn.AddEntry(hist,file_name[file_name.rfind('/')+1:],"f")
            lgn_hist.SetLineWidth(4)
            lgn_hist.SetLineColor(hist.GetLineColor())
            lgn_hist.SetFillColor(hist.GetFillColor())
        return lgn
    elif option == "OFF":
        lgn.Delete()
        return None
    else: return None

def stats(st=None,option="ON",**kwargs):
    if option == "ON":
        st = ROOT.TPaveStats(0.70,0.35,0.89,0.70,"NDC")
        st.SetLineWidth(2)
        st.SetBorderSize(1)
        return st
    elif option == "FUNC":
        st.Clear()
        func = kwargs['func']
        chi = kwargs['chi']
        ndf = kwargs['NDF']
        title = st.AddText("Fit Statistics for %s"%(func.GetName()))
        title.SetTextFont(22)
        st_lines = st.GetListOfLines()
        par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
        par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
        for (name,p) in zip(par_names,par_values):
            entry = ROOT.TLatex(0,0,"%s = %4.4f"%(name,p))
            entry.SetTextFont(132)
            entry.SetTextSize(0.025)
            st_lines.Add(entry)
        if chi is not None:
            entry = ROOT.TLatex(0,0,"c^{2} = %4.4f"%(chi))
            entry.SetTextFont(122)
            entry.SetTextSize(0.025)
            st_lines.Add(entry)
        if ndf is not None:
            entry = ROOT.TLatex(0,0,"NDF = %i"%(ndf))
            entry.SetTextFont(132)
            entry.SetTextSize(0.025)
            st_lines.Add(entry)
        return st
    elif option == "OFF":
        st.Delete()
        return None
    else: return None

if __name__ == "__main__":
    print sys.argv
    print sys.path