import ROOT
import time
import random
import json
import os
import math

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
    'kGreen',
    'kBlue',
    'kCyan',
    'kMagenta',
    'kYellow',
    # 'kWhite',
    'kGray',
    'kBlack',
    'kOrange',
    'kSpring',
    'kTeal',
    'kAzure',
    'kViolet',
    'kPink']

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
    ftr.func.SetNpx(ftr.bins*5)
    func = ftr.func.Clone()

    cmd = " "
    normalized = info['normalized'] == 'True' or info['normalized'] == 'true'
    hasLegend = False
    hasStats = False
    drawfunc = False
    drawhist = False
    drawfhist = False
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
            json_name = "./fit-files/%sfitter-%s-%s-%s.json" % (keyword,ftr.fit_name,ftr.pname,
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
            jpg_name = "./fit-plots/%sfitter-%s-%s-%s.jpg" % (keyword,ftr.fit_name,ftr.pname,
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

    normalized = 'norm' in fit_info_list[0]
    color = kOrange
    cdf_color = None
    animate = False
    hasStack = False
    hasPoint = False
    draw_lines = True
    Npx = ftr.bins
    res = 20
    cmd = " "
    interp_list = []
    interp_append = False
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        if len(q) > 0:
            cmd = q.pop()
        else:
            print ['a','+','c','c.cdf','lines','npx','pt','i-pdf','i-cdf','pdf','cdf','n']
            cmd = raw_input("cmd: ")
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
                color = eval(raw_input("c - pick your color (0-40) or kColor+4-10: "))
        elif cmd == 'c.cdf':
            print "c.cdf - [COLORING ALL CDFS]"
            if len(q) > 0:
                cdf_color = eval(q.pop())
            else:
                cdf_color = eval(raw_input("c.cdf - pick your color (0-40) or kColor+4-10: "))
        elif cmd == 'lines':
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
            if len(q) > 1:
                point = float(q.pop())
                res = int(q.pop())
            else:
                point = float(raw_input("pt - data point for %s: "%(ftr.var)))
                res_str = raw_input("pt - data resolution (num pts): ")
                if res_str is not "":
                    res = int(res_str)
            hasPoint = True
            print "pt - interpolating at mass point %4.3f %s with %i points"%(point,"GeV" if ftr.pname == 'omega' else "MeV",res)
        elif cmd == 'i-pdf':
            print "i-pdf - [INTERPOLATING PDF]"
            print interp_list
            for i_cdf in interp_list:
                # interp_pdf = ROOT.TGraph(i_cdf,'d')
                pass
        elif cmd == 'i-cdf':
            if not hasPoint:
                print "i-cdf - need point: use 'pt' to specify"
            elif not hasStack:
                print "i-cdf - need histograms/functions: use 'pdf or 'cdf' to generate"
            else:
                print "i-cdf - [INTERPOLATING CDF]"
                # canv.Clear()
                # canv.Update()

                cflist = ROOT.TList()

                for num,f in enumerate(flist):
                    # print f
                    # f.Print()
                    cum_fit_name = f.GetName() + "_cdf"
                    cum_func = ROOT.TF1("cum"+str(num)+"_interp"+str(len(interp_list)),fitter.fits[cum_fit_name],ftr.lo,ftr.hi)
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
                    print "i-cdf - ",pars
                    cflist.Add(cum_func)
                
                pcts = []
                for num,cf in enumerate(cflist):
                    x = cf.GetX(0.5)
                    print "i-cdf - mean value for cdf #%i (x): %4.3f"%(num,x)
                    pcts.append(x)

                d = pcts[len(pcts)-1] - pcts[0]
                pct = (point - pcts[0])/d
                print "i-cdf - pct:",pct
                del pcts

                outerfirst = True
                innerfirst = True
                interp_cdf = ROOT.TGraph(res+1)
                interp_cdf.SetLineColor(color+2)
                interp_cdf.SetLineWidth(5)
                interp_cdf.SetNameTitle("interp_cdf_"+str(len(interp_list)),"morphed cdf for %4.3f"%(point))
                for y in [float(i) / res for i in range(res+1)]:
                    print "i-cdf - level: y =",y
                    pts = []
                    for num,cf in enumerate(cflist):
                        if outerfirst and not interp_append:
                            cf.Draw("C SAME" if not innerfirst else "C")
                            canv.Update()
                        x = cf.GetX(y)
                        pts.append(x)
                        print "i-cdf - cdf_%i @ %1.3f = %4.3f (%s %s)"%(
                                num,y,x,ftr.var,"MeV" if ftr.pname == 'phi' else "GeV")
                        innerfirst = False
                    
                    interp_x = pts[0] + pct * (pts[len(pts)-1] - pts[0])
                    print "i-cdf - interpolated point:",interp_x
                    interp_cdf.SetPoint(int(y*res),interp_x,y)
                    
                    if draw_lines:
                        if not outerfirst:
                            del line
                        line = ROOT.TLine(pts[0],y,pts[len(pts)-1],y)
                        line.SetLineColor(color+1)
                        line.SetLineWidth(3 if res < 100 else 2)
                        line.Clone().Draw("SAME")
                        canv.Update()

                    if animate:
                        interp_cdf.Draw("SAME")
                        canv.Update()

                    outerfirst = False
                interp_cdf.SetPoint(res+2,ftr.hi,1.0)
                for i in range(res+2):
                    if math.isnan(interp_cdf.GetPointX(i)):
                        interp_cdf.SetPoint(i,0.,0.)
                    elif i != 0 and interp_cdf.GetPointX(i) == 0:
                        interp_cdf.RemovePoint(i)
                
                interp_list.append(interp_cdf)
                # interp_cdf.GetPoint(0,test_x,test_y)
                interp_cdf.Draw("SAME")
                canv.Update()
                interp_cdf.Print()
        elif cmd == 'pdf':
            print "pdf - [PDFs]"
            hstack = ROOT.THStack("hs","%s of %s for %s"
                    %("Probability Distributions" if normalized else "Event Distributions",ftr.var,ftr.pname))
            flist = ROOT.TList()

            for count,i in enumerate(fit_info_list):
                #set up fitter and fit info
                print "using fit model #%i: %s"%(count,i)
                json_file = open(i,'r')
                info = json.load(json_file)
                file_name = info['file_name']

                ftr = fitter(file_name,fitted=True,fit_info=i)
                ftr.func.SetLineColor(eval(ROOTCOLORS[count]))
                ftr.func.SetLineWidth(5)
                ftr.func.SetNpx(Npx)
                func = ftr.func.Clone()
                flist.Add(func)
                fhist = ROOT.TH1D(func.GetHistogram())
                hstack.Add(fhist)
                json_file.close()
                # func.Print()

            print hstack.GetNhists()
            hstack.GetHists().Print()
            flist.Print()
            first = True
            for f in flist:
                f.Draw("c same" if not first else "c")
                first = False
            # hstack.Draw("hist nostack")
            canv.Update()
            hasStack = True
        elif cmd == 'cdf':
            print "cdf - [CDFs]"
            hstack = ROOT.THStack("hs","%s of %s for %s"
                    %("Cumulative Distributions" if normalized else "Event Counts",ftr.var,ftr.pname))
            flist = ROOT.TList()
            mgraph = ROOT.TMultiGraph()
            mgraph.SetTitle("%s of %s for %s"
                    %("Cumulative Distributions" if normalized else "Event Counts",ftr.var,ftr.pname))
            
            for count,i in enumerate(fit_info_list):
                #set up fitter and fit info
                print "using fit model #%i: %s"%(count,i)
                json_file = open(i,'r')
                info = json.load(json_file)
                file_name = info['file_name']

                ftr = fitter(file_name,fitted=True,fit_info=i)
                if cdf_color is None:
                    ftr.func.SetLineColor(eval(ROOTCOLORS[count]))
                else:
                    ftr.func.SetLineColor(cdf_color)
                ftr.func.SetLineWidth(5)
                ftr.func.SetNpx(Npx)
                func = ftr.func.Clone()
                flist.Add(func)
                mgraph.Add(func.DrawIntegral())
                fhist = ROOT.TH1D(func.GetHistogram().DrawNormalized().GetCumulative())
                hstack.Add(fhist)
                json_file.close()
                # print func
                # print fhist
                # time.sleep(2)
                # canv.Clear()

            # print hstack.GetNhists()
            # hstack.GetHists().Print()
            # flist.Print()
            canv.Clear()
            first = True
            # for num,f in enumerate(flist):
            #     # f.DrawIntegral("ac same" if not first else "ac")
            #     h1 = ROOT.TH1D(f.GetHistogram())
            #     h2 = ROOT.TH1D(h1.DrawNormalized())
            #     for e in canv.GetListOfPrimitives():
            #         if e.GetName() == "Func":
            #             e.Delete()
            #     h3 = h2.GetCumulative()
            #     h3.SetName("h3"+str(num))
            #     h3.Draw("HIST same" if not first else "HIST")
            #     first = False
            #     f.Print()
            #     h3.Print()
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
        # lgn = ROOT.TLegend(0.625,0.785,0.975,0.930)
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

if __name__ == '__main__':
    fit_names = ['gaus','crystalball','landau','landxgaus']
    fit_name = fit_names[2]
    file_name = 'root/TwoProngNtuplizer_etaprime300.root'
    fit_info = './fit-files/fitter-%s-omega-etaprime300.json'%(fit_name)
    
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
            analyze_fit(fit_info)
        elif cmd in ['a.','analyze-all']:
            fit_dir = "./fit-files/"
            dir_list = os.listdir(fit_dir)
            dir_list.sort()
            print dir_list
            for count,f in enumerate(dir_list):
                if f[:4] != "norm":
                    q = deque(['','y','y','s','n','h','c','d','lstats','l'])
                    print "analyzing + normalizing fit for",f
                    print "FILE NUM",count 
                    analyze_fit(fit_dir+f,q)
                    # affirm = raw_input("fit display and data obtained for %s: next file? [y/n]"%(f))
                    affirm = 'y'
                    if affirm != 'y':
                        break
        elif cmd in ['i','interpolate']:
            fit_info_list = [
                    './fit-files/normfitter-crystalball-phi-eta125.json',
                    './fit-files/normfitter-crystalball-phi-eta300.json',
                    './fit-files/normfitter-crystalball-phi-eta500.json',
                    './fit-files/normfitter-crystalball-phi-eta750.json',
                    './fit-files/normfitter-crystalball-phi-eta1000.json',
                    ]
            q = deque(['cdf','kBlack','c.cdf','100','600','pt','500','npx'])
            # q = deque([])
            # name = raw_input("name of normalized fit info file [ENTER or q to EXIT]: ")
            # while name != "" and name != "q" and len(fit_info_list) > 0:
            #     fit_info_list.append(name)
            #     name = raw_input("name of normalized fit info file [ENTER or q to EXIT]: ")
            interpolate_fit(fit_info_list,q)