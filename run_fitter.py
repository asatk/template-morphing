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

def build_fit(file_name,fit_name,fit_info,q=deque()):
    canv = ROOT.TCanvas("canv","title",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    ftr = fitter(file_name,fit_name,fit_info)
    ftr.fit()
    ftr.func.SetLineColor(kPink)
    ftr.func.SetLineWidth(5)
    ftr.func.SetNpx(ftr.bins)
    func = ftr.func.Clone()
    hist = ftr.hist.Clone()
    hist.GetXaxis().SetTitle("%s (GeV)"%(ftr.var))
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
        if len(q) > 0:
            cmd = q.pop()
        else:
            cmd = raw_input("cmd: ")
        if cmd == 's':
            print "[SAVING]"
            if len(q) > 0:
                affirm = q.pop()
            else:
                affirm = raw_input("are you sure you want to SAVE this model? [y/n]")
            if affirm == 'y':
                ftr.jsonify()
        elif cmd == 'r' or cmd == 'refit':
            print "[REFITTING]"
            if len(q) > 0:
                affirm = q.pop()
            else:
                affirm = raw_input("are you sure you want to REFIT? [y/n]")
            if affirm == 'y':
                if len(q) < 3:
                    file_name2 = raw_input("new data: ")
                    fit_name2 = raw_input("new fit function: ")
                    fit_info2 = raw_input("new fit info: ")
                else:
                    file_name2 = q.pop()
                    fit_name2 = q.pop()
                    fit_info = q.pop()

            file_name = file_name if file_name2 == "" else file_name2
            fit_name = fit_name if fit_name2 == "" else fit_name2
            fit_info = fit_info if fit_info2 == "" else fit_info2

            if affirm == 'y':
                canv.Clear()
                ftr = fitter(file_name,fit_name,fit_info)
                ftr.fit()
                ftr.func.SetLineColor(kPink)
                ftr.func.SetLineWidth(5)
                ftr.func.SetNpx(ftr.bins)
                func = ftr.func.Clone()
                hist = ftr.hist.Clone()
                hist.GetXaxis().SetTitle("%s (GeV)"%(ftr.var))
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
        elif cmd == 'rebin':
            print "[REBINNING]"
            if len(q) < 3:
                tempbins = raw_input("new bin size: ")
                templo = raw_input("new lo: ")
                temphi = raw_input("new hi: ")
            else:
                tempbins = q.pop()
                templo = q.pop()
                temphi = q.pop()
            if tempbins is not '':
                ftr.bins = int(tempbins)
            if templo is not '':
                ftr.lo = float(templo)
            if temphi is not '':
                ftr.hi = float(temphi)

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

def interpolate_fit(fit_info_list,q=deque()):
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
    color = kOrange+1
    pdf_color = kBlack
    cdf_color = kBlack
    animate = False
    hasStack = False
    hasPoint = False
    draw_lines = True
    Npx = ftr.bins
    res = 20
    interp_method = "PCT"
    cmd = " "
    interp_cdflist = []
    interp_flist = []
    interp_append = False
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        if len(q) > 0:
            cmd = q.pop()
        else:
            print ['a','+','c','c.pdf','c.cdf','lines','npx','pt','i-pdf','i-cdf','pdf','cdf','n']
            cmd = raw_input("cmd: ")
        c = ROOT.TCanvas()
        if cmd == 'a':
            print "a - %sANIMATING"%("NOT " if animate else "")
            animate = not animate
        elif cmd == '+':
            print "+ - [%sAPPEND INTERPS]"%("DO NOT " if interp_append else "")
            interp_append = not interp_append
        elif cmd == 'c':
            print "c - [PICK COLOR]"
            if len(q) > 0:
                color = eval(q.pop())
            else:
                color = eval(raw_input("c - pick your color (0-40) or kColor+-10: "))
        elif cmd == 'c.pdf':
            print "c.pdf - [COLORING ALL PDFS]"
            if len(q) > 0:
                pdf_color = eval(q.pop())
            else:
                pdf_color = eval(raw_input("c.pdf - pick your color (0-40) or kColor+-10: "))
        elif cmd == 'c.cdf':
            print "c.cdf - [COLORING ALL CDFS]"
            if len(q) > 0:
                cdf_color = eval(q.pop())
            else:
                cdf_color = eval(raw_input("c.cdf - pick your color (0-40) or kColor+-10: "))
        elif cmd == 'lines':
            print "lines - [%sDRAWING LINES]"%("NOT " if draw_lines else "")
            draw_lines = not draw_lines
        elif cmd == 'npx':
            print "pt - [SPECIFY NUMBER OF POINTS IN FUNCTION]"
            if len(q) > 0:
                npx_input = q.pop()
            else:
                npx_input = raw_input("npx - set Npx to: ")
            Npx = int(npx_input) if (npx_input != "" and int(npx_input) <= 10000) else ftr.bins
            print "npx - Npx set to",Npx
        elif cmd == 'pt':
            print "pt - [SPECIFYING POINT OF INTERPOLATION]"
            if len(q) > 2:
                point = float(q.pop())
                res = int(q.pop())
                interp_method = q.pop()
            else:
                point = float(raw_input("pt - data point for %s: "%(ftr.var)))
                res_str = raw_input("pt - data resolution (num pts): ")
                if res_str != "":
                    res = int(res_str)
                interp_method_str = raw_input("pt - choose interpolation method: ['percent/PCT','parameter/PARAM']: ")
                if interp_method_str != "":
                    interp_method = interp_method_str
            hasPoint = True
            print "pt - interpolating at mass point %4.3f %s with %i points"%(point,"GeV" if ftr.pname == 'omega' else "MeV",res)
        elif cmd == 'i-pdf':
            if 'interp_method' not in locals():
                print "i-pdf - must build cdfs from interpolating methods available [i-cdf]"
            else:
                print "i-pdf - [INTERPOLATING PDF]"
                if interp_method == "percent" or interp_method == "PCT":
                    # interp_flist = cdf_derivative(point,interp_flist,interp_cdflist,canv,color=color)
                    cdf_derivative(point,interp_flist,interp_cdflist,canv,color=color)
                elif interp_method == "parameter" or interp_method == "PARAM":
                    for n,interp_f in enumerate(interp_flist):
                        for p in range(interp_f.GetNpar()):
                            print "i-pdf[PARAM] - interp_f#%i param %i %4.3f"%(n,p,interp_f.GetParameter(p))
                        # interp_f.SetParameter(3,1.4)
                        # interp_f.SetParameter(4,5.)
                first = True
                for interp_f in interp_flist:
                    interp_f.Draw(
                            "L" + 
                            ("SAME" if not first else "") +
                            ("A" if interp_method == "percent" or interp_method == "PCT" else ""))
                    canv.Update()
                    first = False
        elif cmd == 'i-cdf':
            if not hasPoint:
                print "i-cdf - need point: use 'pt' to specify"
            elif not hasStack:
                print "i-cdf - need histograms/functions: use 'pdf or 'cdf' to generate"
            else:
                print "i-cdf - [INTERPOLATING CDF - %s]"%(interp_method)
                if interp_method == "percent" or interp_method == "PCT":
                    canv.cd()
                    interp_percent(
                            flist,point,res,Npx,interp_cdflist,interp_append,canv,
                            ftr=ftr,color=color,draw_lines=draw_lines,animate=animate,
                            masspts=masspts)
                    print "post"
                    # for interp_cdf in interp_cdflist:
                    #     interp_cdf.Draw("C" + "SAME" if not first else "")
                    #     canv.Update()
                    #     interp_cdf.Print()
                elif interp_method == "histogram" or interp_method == "HIST":
                    # c = ROOT.TCanvas()
                    interp_hist(fit_info_list,canv,c,ftr=ftr,cdf_color=cdf_color)
                    print "post"
                elif interp_method == "parameter" or interp_method == "PARAM":
                    interp_parameter(flist,masspts,point,interp_flist,canv,ftr=ftr,Npx=Npx)
                    print "post"
                    # first = True
                    # for f in interp_flist:
                    #     f.Draw("C" + "SAME" if not first else "")
                    #     canv.Update()
                    #     f.Print()
                    #     for p in range(f.GetNpar()):
                    #         print f.GetParameter(p)
        elif cmd == 'pdf':
            print "pdf - [PDFs]"
            hstack = ROOT.THStack("hs","%s of %s for %s"
                    %("Probability Distributions" if normalized else "Event Distributions",ftr.var,ftr.pname))
            flist = ROOT.TList()

            for count,i in enumerate(fit_info_list):
                # set up fitter and fit info
                print "using fit model #%i: %s"%(count,i)
                json_file = open(i,'r')
                info = json.load(json_file)
                file_name = info['file_name']

                ftr = fitter(file_name,fitted=True,fit_info=i)
                if pdf_color is None:
                    ftr.func.SetLineColor(ROOTCOLORS[count])
                else:
                    ftr.func.SetLineColor(pdf_color)
                ftr.func.SetLineWidth(5)
                ftr.func.SetNpx(Npx)
                func = ftr.func.Clone()
                flist.Add(func)
                fhist = ROOT.TH1D(func.GetHistogram())
                hstack.Add(fhist)
                json_file.close()

            print hstack.GetNhists()
            hstack.GetHists().Print()
            hstack.Draw("hist nostack")
            canv.Update()
            for interp_f in interp_flist:
                interp_f.Draw("C SAME")
                canv.Update()
            hasStack = True
        elif cmd == 'cdf':
            print "cdf - [CDFs]"
            hstack = ROOT.THStack("hs","%s of %s for %s"
                    %("Cumulative Distributions" if normalized else "Event Counts",ftr.var,ftr.pname))
            flist = ROOT.TList()
            
            for count,i in enumerate(fit_info_list):
                # set up fitter and fit info
                print "using fit model #%i: %s"%(count,i)
                json_file = open(i,'r')
                info = json.load(json_file)
                file_name = info['file_name']

                ftr = fitter(file_name,fitted=True,fit_info=i)
                if cdf_color is None:
                    ftr.func.SetLineColor(ROOTCOLORS[count])
                else:
                    ftr.func.SetLineColor(cdf_color)
                ftr.func.SetLineWidth(5)
                ftr.func.SetNpx(Npx)
                func = ftr.func.Clone()
                flist.Add(func)
                fhist = ROOT.TH1D(func.GetHistogram().DrawNormalized().GetCumulative())
                hstack.Add(fhist)
                json_file.close()
            
            canv.Clear()
            first = True
            hstack.Draw("hist nostack")
            canv.Update()
            hasStack = True
        elif cmd == 'n':
            if not normalized:
                print 'n - [NORMALIZING]'
                for count,name in enumerate(fit_info_list):
                    idx = name.rfind('/')
                    fit_info_list[count] = name[:idx+1] + "norm" + name[idx+1:]
                normalized = True
            else:
                print 'n - [DENORMALIZING]'
                for count,name in enumerate(fit_info_list):
                    idx = name.rfind('/')
                    fit_info_list[count] = name[:idx+1] + name[idx+5:]
                normalized = False

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

def interp_percent(flist,point,res,Npx,interp_cdflist,interp_append,canv,**kwargs):
    cdflist = ROOT.TList()

    ftr = kwargs['ftr']
    lo = ftr.lo
    hi = ftr.hi
    var = ftr.var
    pname = ftr.pname
    color = kwargs['color']
    draw_lines = kwargs['draw_lines']
    animate = kwargs['animate']
    masspts = None if 'masspts' not in kwargs.keys() else kwargs['masspts']

    for num,f in enumerate(flist):
        cum_fit_name = f.GetName() + "_cdf"
        cum_func = ROOT.TF1("cum"+str(num)+"_interp"+str(len(interp_cdflist)),fitter.fits[cum_fit_name],lo,hi)
        cum_func.SetLineColor(f.GetLineColor())
        cum_func.SetLineWidth(f.GetLineWidth())
        cum_func.SetNpx(Npx)
        cum_func.SetParName(0,f.GetParName(0))
        cum_func.SetParameter(0,1.0)
        for i in range(1,f.GetNpar()):
            cum_func.SetParName(i,f.GetParName(i))
            cum_func.SetParameter(i,f.GetParameter(i))
        par_names = list(cum_func.GetParName(i) for i in range(cum_func.GetNpar()))
        par_values = list(cum_func.GetParameter(i) for i in range(cum_func.GetNpar()))
        pars = zip(par_names,par_values)
        print "i-cdf[PCT] - ",pars
        cdflist.Add(cum_func)
    cdflist.Print()
    
    pcts = []
    for num,cdf in enumerate(cdflist):
        x = cdf.GetX(0.5)
        print "i-cdf[PCT] - mean value for cdf #%i (x): %4.3f"%(num,x)
        pcts.append(x)

    idx = 0
    if masspts is not None:
        for jdx,m in enumerate(masspts):
            if point < m:
                idx = jdx
                break
    d = abs(pcts[idx] - pcts[idx-1])
    pct = (point - pcts[idx-1])/d
    print "i-cdf[PCT] - pcts[idx-1],pcts[idx]:",pcts[idx-1],",",pcts[idx]
    print "i-cdf[PCT] - pct:",pct
    del pcts

    outerfirst = True
    innerfirst = True
    interp_cdf = ROOT.TGraph()
    interp_cdf.SetLineColor(color if color is not None else ROOTCOLORS[len(interp_cdflist)])
    interp_cdf.SetLineWidth(5)
    interp_cdf.SetNameTitle("interp_cdf_"+str(len(interp_cdflist)),"morphed cdf for %4.3f"%(point))
    interp_cdf.SetPoint(0,0.,0.)

    # get x values for both interpolating cdfs from equal y-values at some step determined by Npx
    for y in [float(i) / res for i in range(res+1)]:
        print "i-cdf[PCT] - step: %i ; level: y=%4.3f"%(int(round(y*res)),y)
        pts = []
        for num,cdf in enumerate(cdflist[idx-1:idx+1]):
            if outerfirst and draw_lines:
                cdf.Draw("C SAME" if interp_append or not innerfirst else "C")
                canv.Update()
            x = cdf.GetX(y)
            # if math.isnan(x):
            #     x = 0
            pts.append(x)
            print "i-cdf[PCT] - cdf_%i @ %1.3f = %4.3f (%s %s)"%(
                    num+idx,y,x,var,"MeV" if pname == 'phi' else "GeV")
            innerfirst = False
        
        interp_x = pts[0] + pct * (pts[1] - pts[0])
        interp_cdf.SetPoint(int(round(y*res)+1),interp_x,y)
        
        if draw_lines:
            if not outerfirst:
                del line
            line = ROOT.TLine(pts[0],y,pts[len(pts)-1],y)
            line.SetLineColor(color-1 if color is not None else ROOTCOLORS[len(interp_cdflist)])
            line.SetLineWidth(3 if res < 100 else 2)
            line.Clone().Draw("SAME")
            canv.Update()

        if animate:
            interp_cdf.Draw("SAME")
            canv.Update()

        outerfirst = False
    
    remove_pts = []
    if math.isnan(interp_cdf.GetPointX(0)):
        interp_cdf.SetPoint(0,0.,0.)
    for i in range(res+1):
        p = interp_cdf.GetPointX(i)
        # print i,p
        if i != 0 and (math.isnan(p) or p <= 0):
            remove_pts.append(i)

    remove_count = 0
    for r_n,r_pt in enumerate(remove_pts):
        print "removed",r_pt
        interp_cdf.RemovePoint(r_pt-r_n)
    interp_cdf.SetPoint(interp_cdf.GetN(),hi,1.0)
    interp_cdflist.append(interp_cdf)
    interp_cdf.Draw("SAME")
    canv.Update()
    cdflist[idx-1].Print()
    cdflist[idx].Print()
    # interp_cdf.Print()

def interp_hist(fit_info_list,point,canv,c,**kwargs):
    # canv.Clear()
    # c = ROOT.TCanvas()
    ftr = kwargs['ftr']
    cdf_color = kwargs['cdf_color']
    fit_name = ftr.fit_name
    # chain = ROOT.TChain("twoprongNtuplizer/fTree")

    hstack = ROOT.THStack("hs","%s of %s for %s"
            %("Cumulative Distributions",ftr.var,ftr.pname))

    first = True
    for (num,f) in enumerate(fit_info_list):
        with open(f) as json_file:
            info = json.load(json_file)

            exec('hist'+str(num)+' = ROOT.TH1D("hist"+str(num),"%s cdf - %s"%(ftr.fit_name,ftr.var),ftr.bins,ftr.lo,ftr.hi)')

            hist = eval('hist'+str(num))

            x_ax = hist.GetXaxis()
            if ftr.pname == 'phi':
                x_ax.SetTitle("%s (MeV)"%(ftr.var))
            else:
                x_ax.SetTitle("%s (GeV)"%(ftr.var))
            x_ax.CenterTitle(True)
            y_ax = hist.GetYaxis()
            y_ax.SetTitle("Event Probability Density")
            y_ax.CenterTitle(True)
            hist.SetMaximum(250)

            if cdf_color is None:
                hist.SetLineColor(ROOTCOLORS[num])
                hist.SetLineColor(ROOTCOLORS[num]+1)
            else:
                hist.SetLineColor(cdf_color)
                hist.SetLineColor(cdf_color+1)

            hist.SetLineWidth(3)
            hist.SetStats(0)

            chain = ROOT.TChain("twoprongNtuplizer/fTree")
            chain.Add(info['file_name'])
            draw_s = info['var'] + ">>hist"+str(num)
            cut_s = info['cuts']
            chain.Draw(draw_s, cut_s, option="goff")
            # chain.Draw(draw_s, cut_s, option="HIST" + ("" if first else "SAME"))
            hist = hist.DrawNormalized().GetCumulative()
            hist.SetMaximum(1.05)
            hist.Draw("HIST")
            # hist.Draw("HIST" + ("" if first else "SAME"))
            hstack.Add(hist.Clone())
            c.Update()
            first = False
            json_file.close()
    hstack.Print()
    # canv.Clear()
    canv.cd()
    hstack.Clone().Draw("hist nostack same")
    canv.Update()
    # c.Update()

    pcts = []
    for num,cdf_hist in enumerate(hstack):
        cdf = ROOT.TGraph(cdf_hist)
        # cdf.Print()
        x = cdf.GetPointY(cdf.GetN()/2)
        print "i-cdf[PCT] - mean value for cdf #%i (x): %4.3f"%(num,x)
        pcts.append(x)

    idx = 0
    if masspts is not None:
        for jdx,m in enumerate(masspts):
            if point < m:
                idx = jdx
                break
    d = abs(pcts[idx] - pcts[idx-1])
    pct = (point - pcts[idx-1])/d
    print "i-cdf[PCT] - pcts[idx-1],pcts[idx]:",pcts[idx-1],",",pcts[idx]
    print "i-cdf[PCT] - pct:",pct
    del pcts

    outerfirst = True
    innerfirst = True
    interp_cdf = ROOT.TGraph()
    interp_cdf.SetLineColor(color if color is not None else ROOTCOLORS[len(interp_cdflist)])
    interp_cdf.SetLineWidth(5)
    interp_cdf.SetNameTitle("interp_cdf_"+str(len(interp_cdflist)),"morphed cdf for %4.3f"%(point))
    interp_cdf.SetPoint(0,0.,0.)

    # get x values for both interpolating cdfs from equal y-values at some step determined by Npx
    for y in [float(i) / res for i in range(res+1)]:
        print "i-cdf[PCT] - step: %i ; level: y=%4.3f"%(int(round(y*res)),y)
        pts = []
        for num,cdf in enumerate(cdflist[idx-1:idx+1]):
            if outerfirst and draw_lines:
                cdf.Draw("C SAME" if interp_append or not innerfirst else "C")
                canv.Update()
            x = cdf.GetPointY(y)
            # if math.isnan(x):
            #     x = 0
            pts.append(x)
            print "i-cdf[PCT] - cdf_%i @ %1.3f = %4.3f (%s %s)"%(
                    num+idx,y,x,var,"MeV" if pname == 'phi' else "GeV")
            innerfirst = False
        
        interp_x = pts[0] + pct * (pts[1] - pts[0])
        interp_cdf.SetPoint(int(round(y*res)+1),interp_x,y)
        
        if draw_lines:
            if not outerfirst:
                del line
            line = ROOT.TLine(pts[0],y,pts[len(pts)-1],y)
            line.SetLineColor(color-1 if color is not None else ROOTCOLORS[len(interp_cdflist)])
            line.SetLineWidth(3 if res < 100 else 2)
            line.Clone().Draw("SAME")
            canv.Update()

        if animate:
            interp_cdf.Draw("SAME")
            canv.Update()

        outerfirst = False
    
    remove_pts = []
    if math.isnan(interp_cdf.GetPointX(0)):
        interp_cdf.SetPoint(0,0.,0.)
    for i in range(res+1):
        p = interp_cdf.GetPointX(i)
        # print i,p
        if i != 0 and (math.isnan(p) or p <= 0):
            remove_pts.append(i)

    remove_count = 0
    for r_n,r_pt in enumerate(remove_pts):
        print "removed",r_pt
        interp_cdf.RemovePoint(r_pt-r_n)
    interp_cdf.SetPoint(interp_cdf.GetN(),hi,1.0)
    interp_cdflist.append(interp_cdf)
    interp_cdf.Draw("SAME")
    canv.Update()
    cdflist[idx-1].Print()
    cdflist[idx].Print()
    # interp_cdf.Print()

def interp_parameter(flist,masspts,point,interp_flist,canv,**kwargs):
    print "i-cdf[PARAM] - [INTERPOLATING PARAMETERS WITH OLS]"
    
    param_flist = []
    interp_params = []
    mg = ROOT.TMultiGraph()
    ftr = kwargs['ftr']
    Npx = kwargs['Npx']
    # iterate through parameters - make one fit per parameter for all functions
    for i in range(flist[0].GetNpar()):
        g = ROOT.TGraph()
        g.SetLineWidth(5)
        g.SetLineColor(ROOTCOLORS[i+5])
        g.SetNameTitle("OLS - %s#%i"%(flist[0].GetParameter(i),i))


        par_names = list(flist[i].GetParName(j) for j in range(flist[i].GetNpar()))
        par_values = list(flist[i].GetParameter(j) for j in range(flist[i].GetNpar()))
        pars = zip(par_names,par_values)
        print "i-cdf[PCT] - ",pars

        pts = {}
        # gather each fn's parameter value for current parameter
        for num,f in enumerate(flist):
            print "i-cdf[PARAM] - %s#%i: %s = %6.4f"%(f.GetName(),num,f.GetParName(i),f.GetParameter(i))
            pts[num] = f.GetParameter(i)
            g.SetPoint(num,masspts[num],f.GetParameter(i))
        g.Print()
        pts = sorted(pts.items(), key = lambda kv:(kv[1],kv[0]))
        idxs = [idx[0] for idx in pts]
        pts = [pt[1] for pt in pts]
        q1 = pts[int(math.floor(len(pts)*0.25))]
        q3 = pts[int(math.floor(len(pts)*0.75))]
        iqr = q3 - q1
        if pts[0] < q1 - 1.5*(iqr):
            print "i-cdf[PARAM] - EXCLUDING POINT %4.3f from MASS %4.3f %s"%(
                    pts[0],masspts[num],"MeV" if ftr.pname == 'phi' else "GeV")
            g.RemovePoint(idxs[0])
            g.Print()
        elif pts[len(pts)-1] > q3 + 1.5*(iqr):
            print "i-cdf[PARAM] - EXCLUDING POINT %4.3f from MASS %4.3f %s"%(
                    pts[len(pts)-1],masspts[num],"MeV" if ftr.pname == 'phi' else "GeV")
            g.RemovePoint(idxs[len(idxs)-1])
            g.Print()
        param_f = ROOT.TF1("OLS#%i"%(i),"pol1",ftr.lo,ftr.hi)
        param_f.SetLineColor(kRed)
        param_f.SetLineWidth(3)
        g.Fit(param_f,"SM0Q rob=0.8")
        param_OLS = g.GetFunction("OLS#%i"%(i))
        param_flist.append(param_OLS)
        interp_params.append(param_OLS.Eval(point))
        mg.Add(g)

    mg.Draw("AL")
    canv.Update()
    interp_f = ROOT.TF1("param interp",str(flist[0].GetExpFormula()),ftr.lo,ftr.hi)
    interp_f.SetLineWidth(5)

    print "i-cdf[PARAM] - formula",interp_f.GetExpFormula()
    for j in range(interp_f.GetNpar()):
        print "i-cdf[PARAM] - param",j,interp_params[j]
        interp_f.SetParameter(flist[0].GetParName(j),interp_params[j])
        param_flist[j].Draw("SAME")
        canv.Update()

    interp_f.SetNpx(Npx)
    print "i-cdf[PARAM] - interp_f integral",interp_f.GetHistogram().Integral()
    print "i-cdf[PARAM] - norm factor",(1. / interp_f.GetHistogram().Integral())
    interp_f.SetParameter('Constant',interp_f.GetParameter('Constant') * 1. / interp_f.GetHistogram().Integral())
    print "i-cdf[PARAM] - integral - should be 1.0:",interp_f.GetHistogram().Integral()

    interp_flist.append(interp_f)
    ROOT.gDirectory.Append(mg)

def cdf_derivative(point,interp_flist,interp_cdflist,canv,**kwargs):
    color = kwargs['color']
    # canv.Clear()
    interp_mg = ROOT.TMultiGraph()
    num = 0
    interp_cdf = interp_cdflist[0]
    # for num,interp_cdf in enumerate(interp_cdflist):
    interp_f = ROOT.TGraph()
    interp_f.SetNameTitle("interp_f_"+str(num),"morphed func for %4.3f"%(point))
    interp_f.SetLineColor(color if color is not None else ROOTCOLORS[num])
    interp_f.SetLineWidth(5)
    for i in range(interp_cdf.GetN()-1):
        dydx = (interp_cdf.GetPointY(i+1)-interp_cdf.GetPointY(i))/(interp_cdf.GetPointX(i+1)-interp_cdf.GetPointX(i))
        print "i-pdf[PCT] - dydx[%i]=%2.4f"%(i,dydx)
        interp_f.SetPoint(i,interp_cdf.GetPointX(i),dydx)
    interp_f.SetPoint(interp_cdf.GetN()-1,interp_cdf.GetPointX(interp_cdf.GetN()-1),0.)
    print "INTERP_F#"+str(num)
    interp_f.Print()
    print "INTERP EVAL1a",interp_f.Eval(interp_f.GetPointX(1)/2)
    print "INTERP EVAL1b",interp_f.Eval(interp_f.GetPointX(1)/2,0,"")
    print "INTERP EVAL2",interp_f.Eval(interp_f.GetPointX(1)/2,0,"S")
    temp_res = 200
    end_pt = interp_f.GetPointX(1)
    for i in range(temp_res):
        pt = i*end_pt/temp_res
        print "~~~PT - %4.2f~~~"%(pt)
        print "INTERP EVAL1b",interp_f.Eval(pt,0,"")
        print "INTERP EVAL2",interp_f.Eval(pt,0,"S")
    # print interp_f.Integral(0,interp_cdf.GetN())
    interp_flist.append(interp_f)
    # interp_flist.Add(interp_f)
    # interp_mg.Add(interp_f)

    # interp_mg.Draw("AC")
    return interp_flist

def view_fits(fit_info_list,opt="HIST",q=deque()):
    
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
    pdf_color = 5
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
        if pdf_color == 5:
            ftr.func.SetLineColor((ROOTCOLORS2[count%5] + (-4 if count > 4 else 2)))
        elif pdf_color == 2:
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
        hstack.GetHists().Print()
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
    if True:
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


if __name__ == '__main__':
    fit_names = ['gaus','crystalball','landau','landxgaus']
    ptcl_names = ['phi', 'omega']
    fit_name = fit_names[3]
    ptcl = ptcl_names[1]
    # file_name = 'root/TwoProngNtuplizer_eta500.root'
    file_name = 'root/TwoProngNtuplizer_eta750.root'
    # fit_info = './fit-files/fitter-%s-%s-eta0500.json'%(fit_name,ptcl)
    fit_info = './fit-files/fitter-%s-%s-etaprime0750.json'%(fit_name,ptcl)
    
    cmd = " "
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        cmd = raw_input("run:")
        if cmd in ['b','build']:
            fit_info = 'fitter-init.json'
            build_fit(file_name,fit_name,fit_info)
        elif cmd in ['b.','build-all']:
            file_dir = "./root/"
            for f in os.listdir(file_dir):
                for fn in fit_names:
                    q = deque(['','y','s'])
                    print "building %s fit for %s"%(fn.upper(),f)
                    build_fit(file_dir+f,fn,'fitter_init.json',q)
                    # affirm = raw_input("%s fit for %s: next FIT? [y/n]"%(fn.upper(),f))
                    affirm = 'y'
                    if affirm != 'y':
                        break
                # affirm2 = raw_input("%s fit for %s: next FILE? [y/n]"%(fn.upper(),f))
                affirm2 = 'y'
                if affirm2 != 'y':
                    break
        elif cmd in ['a','analyze']:
            # q = deque(['','y','y','s','n','h','c','d','lstats','l'])
            q = deque(['n','h','c hist','d','lstats','l'])
            analyze_fit(fit_info, q=q)
        elif cmd in ['a.','analyze-all']:
            fit_dir = "./fit-files/"
            dir_list = os.listdir(fit_dir)
            print dir_list
            dir_list.sort()
            print dir_list
            for count,f in enumerate(dir_list):
                if f[:4] != "norm":
                    # q = deque(['','n','h','c hist','d','lstats','l'])
                    q = deque(['n','h','c hist','d','lstats','l'])
                    # q = deque(['','y','y','s','n','h','c','d','lstats','l'])
                    print "analyzing + normalizing fit for",f
                    print "FILE NUM",count 
                    analyze_fit(fit_dir+f,q)
                    # affirm = raw_input("fit display and data obtained for %s: next file? [y/n]"%(f))
                    affirm = 'y'
                    if affirm != 'y':
                        break
                    # time.sleep(1)
        elif cmd in ['i','interpolate']:
            fit_info_list = [
                    './fit-files/normfitter-%s-phi-eta0125.json'%(fit_name),
                    './fit-files/normfitter-%s-phi-eta0300.json'%(fit_name),
                    './fit-files/normfitter-%s-phi-eta0500.json'%(fit_name),
                    './fit-files/normfitter-%s-phi-eta0750.json'%(fit_name),
                    './fit-files/normfitter-%s-phi-eta1000.json'%(fit_name)
                    ]
            q = deque(['i-cdf','cdf','lines','PARAM','500','500','pt','500','npx'])
            # q = deque(['i-cdf','cdf','lines','HIST','1000','500','pt','1000','npx'])
            # q = deque([])
            # name = raw_input("name of normalized fit info file [ENTER or q to EXIT]: ")
            # while name != "" and name != "q" and len(fit_info_list) > 0:
            #     fit_info_list.append(name)
            #     name = raw_input("name of normalized fit info file [ENTER or q to EXIT]: ")
            interpolate_fit(fit_info_list,q)
        elif cmd in ['v','view']:
            fit_info_list_view = [
                    './fit-files/normfitter-%s-%s-eta0125.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-eta0300.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-eta0500.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-eta0750.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-eta1000.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime0125.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime0300.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime0500.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime0750.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime1000.json'%(fit_name,ptcl)
                    ]
            view_fits(fit_info_list_view,"HIST")
        elif cmd in ['l','list']:
            fit_info_list_list = [
                    './fit-files/normfitter-%s-%s-eta0125.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-eta0300.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-eta0500.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-eta0750.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-eta1000.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime0125.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime0300.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime0500.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime0750.json'%(fit_name,ptcl),
                    './fit-files/normfitter-%s-%s-etaprime1000.json'%(fit_name,ptcl)
                    ]
            list_pars(fit_info_list_list)