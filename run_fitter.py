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
    ftr.func.SetNpx(ftr.bins)
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
            print ['s','d','l','lstats','i','c','n','h','p','json-p','+','*','/','cmd-q']
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
            
            if drawhist:
                hist.Draw("HIST SAME")

            if "same" not in draw_opts or "SAME" not in draw_opts:
                drawhist = False
                drawcum = False
            if ("hist" not in draw_opts and "HIST" not in draw_opts)\
                    and ("c" in draw_opts and "C" not in draw_opts):
                drawfhist = False
                drawfunc = True
            else: 
                drawfhist = True
                drawfunc = False
            func.Draw(draw_opts)
            canv.Update()
            if hasLegend:
                lgn = legend(lgn,"FUNC",func=func,fit_name=ftr.fit_name)
                lgn.Draw()
                canv.Update()
            if hasStats:
                st = stats(st,"FUNC",func=func,chi=ftr.chi)
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
                    st = stats(st,"FUNC",chi=ftr.chi,func=func)
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
            if drawhist:
                hist.GetCumulative().Draw()
                print "c - hist Integral",hist.Integral()
            if drawfhist:
                func.GetHistogram().GetCumulative().Draw("SAME" if drawhist else "")
                print "c - func histogram Integral",func.GetHistogram().Integral()
            canv.Update()
            drawcum = True
        elif cmd == 'n':
            if not normalized:
                print "n - [NORMALIZING]"
                norm_factor = 1. / func.GetHistogram().Integral()
                func.SetParameter(0,func.GetParameter(0)*norm_factor)
                if drawhist:
                    hist.GetYaxis().SetTitle("Event Probability Density")
                    hist = ROOT.TH1D(hist.DrawNormalized("HIST"))
                    canv.Update()
                    if drawfunc or drawfhist:
                        # y_ax_max = float(1.05*max([func.GetMaximum(),hist.GetMaximum()]))
                        # hist.SetMaximum(y_ax_max)
                        # hist.Draw("HIST")
                        pass
                if drawfhist:
                    # func.SetMaximum(y_ax_max)
                    # func.GetHistogram().DrawNormalized("HIST"+("SAME" if drawhist else ""))
                    # canv.Update()
                    pass
                if drawfunc:
                    # func.SetMaximum(y_ax_max)
                    # func.Draw("C"+("SAME" if drawfhist or drawhist else ""))
                    # canv.Update()
                    pass
                normalized = True
            elif normalized:
                print "n - [DE-NORMALIZING]"
                norm_factor = 1. if 'norm_factor' not in locals() else 1. / norm_factor
                func.SetParameter(0,func.GetParameter(0)*norm_factor)
                if drawhist:
                    hist.GetYaxis().SetTitle("Events")
                    hist.Scale(norm_factor)
                    if drawfunc or drawfhist:
                        # hist.SetMaximum(float(1.05*max([func.GetMaximum(),hist.GetMaximum()])))
                        pass
                    hist.Draw("HIST")
                if drawfhist:
                    func.Draw("HIST"+("SAME" if drawhist else ""))
                    canv.Update()
                if drawfunc:
                    func.Draw("C"+("SAME" if drawhist or drawfhist else ""))
                    canv.Update()
                normalized = False
            
            if hasLegend:
                lgn = legend(lgn,"HIST",hist=hist,file_name=ftr.file_name)
                lgn.Draw()
                canv.Update()
            if hasStats:
                st = stats(st,option="FUNC",func=func,chi=ftr.chi)
                st.Draw()
                canv.Update()

            print "n - normalization factor: ",norm_factor
            canv.Update()
        elif cmd == 'h':
            print "h - [HISTOGRAM PLOTTING]"
            if drawhist:
                hist.Delete()
                chain.Delete()
            drawhist = True
            hist = ROOT.TH1D("hist","%s fit - %s"%(ftr.fit_name,ftr.var),ftr.bins,ftr.lo,ftr.hi)

            x_ax = hist.GetXaxis()
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
                # hist.SetMaximum(float(1.05*max([func.GetMaximum(),hist.GetMaximum()])))
                pass
            canv.Update()

            if normalized:
                hist = ROOT.TH1D(hist.DrawNormalized("HIST"))
                canv.Update()

            if drawfhist:
                func.Draw("HIST SAME")
            if drawfunc:
                func.Draw("C SAME")
            if hasLegend:
                lgn = legend(lgn,"HIST",hist=hist,file_name=ftr.file_name)
                lgn.Draw()
                canv.Update()
            if hasStats:
                st.Draw()
                canv.Update()

            print "h - hist intg",hist.Integral()
            print "h - fhist intg",func.GetHistogram().Integral()
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
    first = True
    print fit_info_list

    for i in fit_info_list:
        #set up fitter and fit info
        print "using file:",i
        json_file = open(i,'r')
        info = json.load(json_file)
        file_name = info['file_name']
        ftr = fitter(file_name,fitted=True,fit_info=i)
        ftr.func.SetLineColor(eval(ROOTCOLORS[random.randint(0,len(ROOTCOLORS)-1)]))
        ftr.func.SetLineWidth(5)
        ftr.func.SetNpx(ftr.bins)
        func = ftr.func.Clone()
        func.Draw("hist" + ("" if first else "same"))
        canv.Update()
        first = False
    
    cmd = " "
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        if len(q) > 0:
            cmd = q.pop()
        else:
            print ['s','d','l','i','c','n','json-p','fhist','h','p','+','*','/','cmd-q']
            cmd = raw_input("cmd: ")
        if cmd == 'pt':
            print "pt - [SPECIFYING POINT]"
            point = float(raw_input("pt - data point for",ftr.var))

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
        entry = ROOT.TLatex(0,0,"c^{2} = %4.4f"%(chi))
        entry.SetTextFont(122)
        entry.SetTextSize(0.025)
        st_lines.Add(entry)
        return st
    elif option == "OFF":
        st.Delete()
        return None
    else: return None

if __name__ == '__main__':
    fit_names = ['gaus','crystalball','landau','landxgaus']
    fit_name = fit_names[3]
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
            for f in os.listdir(fit_dir):
                if f[:4] != "norm":
                    q = deque(['','y','y','s','n','h','c','d','lstats','l'])
                    print "analyzing + normalizing fit for",f
                    analyze_fit(fit_dir+f,q)
                    # affirm = raw_input("fit display and data obtained for %s: next file? [y/n]"%(f))
                    affirm = 'y'
                    if affirm != 'y':
                        break
        elif cmd in ['i','interpolate']:
            fit_info_list = [
                    './fit-files/normfitter-landxgaus-omega-etaprime500.json',
                    './fit-files/normfitter-landxgaus-omega-etaprime750.json'
                    ]
            # name = raw_input("name of normalized fit info file [ENTER or q to EXIT]: ")
            # while name != "" and name != "q" and len(fit_info_list) > 0:
            #     fit_info_list.append(name)
            #     name = raw_input("name of normalized fit info file [ENTER or q to EXIT]: ")
            interpolate_fit(fit_info_list)