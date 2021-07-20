import ROOT
import time
import json
import sys
import re

from collections import deque

from ROOT import Math
from ROOT import TMath

from ROOT import kRed
from ROOT import kGreen 
from ROOT import kBlue  
from ROOT import kMagenta 
from ROOT import kCyan  
from ROOT import kAzure 
from ROOT import kPink

from build.fitter import fitter

class analyzer:

    def __init__(self,fit_info,q=None):
        self.q = deque() if q is None else q

        #set up fitter
        self.ftr = fitter(fit_info)

        #set up dictionary holding drawing parameters
        self.drawing_params = {
            'legend':False,
            'stats':False,
            'func':False,
            'hist':False,
            'fhist':False,
            'cum':False,
            'norm':False
        }

        #set up canvas
        self.canv = ROOT.TCanvas("canv","title",1200,900)
        self.canv.DrawCrosshair()

    def analyze(self):
        self.canv.cd()
        
        #set up function to be analyzed
        self.func = self.ftr.func.Clone()
        self.func.SetLineColor(kPink)
        self.func.SetLineWidth(5)
        # self.func.SetNpx(ftr.bins*5)
        self.func.SetNpx(self.ftr.bins)

        #set up data histogram paired with function
        self.hist = ROOT.TH1D("hist","%s fit - %s"%(self.ftr.fit_name,self.ftr.var),self.ftr.bins,self.ftr.lo,self.ftr.hi)

        cmd = " "

        while not cmd == "" :
            if len(self.q) > 0:
                cmd = self.q.pop()
            else:
                print ['s','d','l','lstats','i','c','n','h','line','p','json','+','*','/','cmd-q']
                cmd = raw_input("cmd: ")
            if cmd == 's':
                print "s - [SAVING]"
                self.__save()
            elif cmd == 'd':
                print "d - [DRAWING]"
                self.__draw()
            elif cmd == 'l':
                print "l - [LEGEND]"
                if self.drawing_params['func'] or self.drawing_params['fhist']:
                    self.__legend(func=self.func,chi=self.ftr.chi)
                elif self.drawing_params['hist']:
                    self.__legend(hist=self.hist,run_file=self.ftr.run_file)
                self.canv.Update()
            elif cmd == 'lstats':
                print "lstats - [STATS BOX]"
                self.__stats(func=self.func,chi=self.ftr.chi,NDF=self.ftr.NDF)
                self.canv.Update()
            elif cmd == 'i':
                print "i - [INTEGRATING]"
                self.__integrate()
            elif cmd == 'c':
                print "c - [CUMULATIVE]"
                self.__cum()
            elif cmd == 'n':
                self.__norm()
            elif cmd == 'h':
                print "h - [HISTOGRAM PLOTTING]"
                self.__hist()
            elif cmd == 'line':
                print "line - [LINE AT MEAN]"
                self.__line()
            elif cmd == 'p':
                print "p - [FETCHING PARAMETERS]"
                self.__pars()
            elif cmd == 'json':
                print "json-p - [PRINTING STATS]"
                self.__json_dump()
            elif cmd == '+':
                print "+ - [ADDING FILE, NORMALIZING]"
                self.__add()
            elif cmd == "*":
                print "* - [RESTARTING]"
                self.__new()
            elif cmd == '/':
                print "/ - [CLEARING]"
                self.__clear()            
            elif cmd == 'cmd-q':
                print "cmd-q - [COMMAND QUEUE]"
                self.__fill_q()
                print "cmd-q - queue:",self.q

    #CMD:"s"
    def __save(self):
        if self.drawing_params['norm']: keyword = "norm"
        elif self.drawing_params['cum']: keyword = "cum"
        else: keyword = ""
        json_name = "../out/fits/%sfitter-%s-%s-%s.json" % (keyword,self.ftr.fit_name,self.ftr.pname,
                self.ftr.run_file[eta_start:eta_start + num_start] + "%04i"%(num))
        if len(self.q) > 0:
            affirm1 = self.q.pop()
        else:
            affirm1 = raw_input("s - are you sure you want to SAVE this MODEL?\n\
                    will save as %s :[y/n]"%(json_name))
        if affirm1 == 'y':
            if self.ftr.normalized:
                self.ftr.func.SetParameter(0,self.ftr.func.GetParameter(0))
            self.ftr.jsonify(fit_info=json_name)
        jpg_name = "../out/plots/%sfitter-%s-%s-%s.jpg" % (keyword,self.ftr.fit_name,self.ftr.pname,
                self.ftr.run_file[eta_start:eta_start + num_start] + "%04i"%(num))
        if len(self.q) > 0:
            affirm2 = self.q.pop()
        else:
            affirm2 = raw_input("s - are you sure you want to SAVE this IMAGE?\n\
                    will save as %s :[y/n]"%(jpg_name))
        if affirm2 == 'y':
            self.canv.SaveAs(jpg_name)

    #CMD:"d"
    def __draw(self):
        if len(self.q) > 0:
            draw_opts = self.q.pop()
        else:
            draw_opts = raw_input("draw option input string [SAME, L, C, FC, HIST]: ")
        
        if not("same" in draw_opts or "SAME" in draw_opts):
            self.drawing_params['cum'] = False
            self.canv.Clear()
        if self.drawing_params['hist']:
            self.hist.Draw("HIST")
            self.canv.Update()
        if len(draw_opts) == 0\
            or "l" in draw_opts or "L" in draw_opts\
            or "c" in draw_opts or "C" in draw_opts\
            or "fc" in draw_opts or "FC" in draw_opts:
            self.drawing_params['func'] = True
        else:
            self.drawing_params['func'] = False
            if "hist" in draw_opts or "HIST" in draw_opts:
                self.drawing_params['fhist'] = True

        self.ftr.func.Draw(draw_opts + ("SAME" if self.drawing_params['hist'] else ""))
        self.canv.Update()
        if self.drawing_params['legend']:
            self.lgn.list_funcs(self.ftr.func,self.ftr.fit_name)
            self.lgn.draw()
            self.canv.Update()
        if self.drawing_params['stats']:
            self.st.list_funcs(self.ftr.func,chi=self.ftr.chi,NDF=self.ftr.NDF)
            self.st.draw()
            self.canv.Update()

    class legend:
        """legend class used in analyzer graphics"""
        def __init__(self):
            self.lgn = ROOT.TLegend(0.55,0.75,0.89,0.88)
            self.__setup()

        def __delete__(self,instance):
            self.lgn.Delete()

        def __setup(self):
            self.lgn.SetTextFont(22)
            self.lgn.SetHeader("Fit Function and Histogram")
            self.lgn.SetTextFont(132)
            self.lgn.SetEntrySeparation(0.05)
            self.lgn.SetTextSize(0.025)
            self.lgn.SetLineWidth(2)
            self.lgn.SetFillColor(19)

        def list_funcs(self,func,fit_name):
            lgn_entry_list = self.lgn.GetListOfPrimitives()
            entry_printed = False
            for obj in lgn_entry_list:
                entry = ROOT.TLegendEntry(obj)
                if entry.GetLabel() == "%s fit curve"%(fit_name):
                    entry_printed = True
            if not entry_printed:
                lgn_func = lgn.AddEntry(func,"%s fit curve"%(fit_name),"l")
                lgn_func.SetLineWidth(4)
                lgn_func.SetLineColor(func.GetLineColor())

        def list_hists(self,hist,run_file):
            lgn_entry_list = lgn.GetListOfPrimitives()
            entry_printed = False
            for obj in lgn_entry_list:
                entry = ROOT.TLegendEntry(obj)
                if entry.GetLabel() == run_file[run_file.rfind('/')+1:]:
                    entry_printed = True
            if not entry_printed:
                lgn_hist = lgn.AddEntry(hist,run_file[run_file.rfind('/')+1:],"f")
                lgn_hist.SetLineWidth(4)
                lgn_hist.SetLineColor(hist.GetLineColor())
                lgn_hist.SetFillColor(hist.GetFillColor())

        def draw(self,opt=""):
            self.lgn.Draw(opt)

    #CMD:"l"
    def __legend(self,**kwargs):
        if not self.drawing_params['legend']:
            self.drawing_params['legend'] = True
            self.lgn = legend()
            self.lgn.draw()
            self.canv.Update()
            if self.drawing_params['func'] or self.drawing_params['fhist']:
                self.lgn.list_funcs(kwargs['func'],kwargs['fit_name'])
                self.lgn.draw()
                self.canv.Update()
            if self.drawing_params['hist']:
                self.lgn.list_hists(kwargs['hist'],kwargs['run_file'])
                self.lgn.Draw()
                self.canv.Update()
        else:
            self.drawing_params['legend'] = False
            del self.lgn
            self.canv.Update()

    class statsbox:
        """statsbox class used in analyzer"""
        def __init__(self):
            self.st = ROOT.TPaveStats(0.70,0.35,0.89,0.70,"NDC")
            self.__setup()

        def __delete__(self,instance):
            self.st.Delete()

        def __setup(self):
            st.SetLineWidth(2)
            st.SetBorderSize(1)

        def list_funcs(self,func,**kwargs):
            st.Clear()
            chi = kwargs['chi']
            NDF = kwargs['NDF']
            # create title
            title = st.AddText("Fit Statistics for %s"%(func.GetName()))
            title.SetTextFont(22)
            st_lines = st.GetListOfLines()
            
            # make param/value entries
            par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
            par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
            for (name,p) in zip(par_names,par_values):
                entry = ROOT.TLatex(0,0,"%s = %4.4f"%(name,p))
                entry.SetTextFont(132)
                entry.SetTextSize(0.025)
                st_lines.Add(entry)
            
            # make chi entry
            if chi is not None:
                entry = ROOT.TLatex(0,0,"c^{2} = %4.4f"%(chi))
                entry.SetTextFont(122)
                entry.SetTextSize(0.025)
                st_lines.Add(entry)
            
            # make NDF entry
            if NDF is not None:
                entry = ROOT.TLatex(0,0,"NDF = %i"%(NDF))
                entry.SetTextFont(132)
                entry.SetTextSize(0.025)
                st_lines.Add(entry)

        def draw(self,opt=""):
            self.st.Draw(opt)

    #CMD:"lstats"
    def __stats(self,**kwargs):
        if not self.drawing_params['stats']:
            self.drawing_params['stats'] = True
            self.st = statsbox()
            self.st.draw()
            self.canv.Update()
            if self.drawing_params['func'] or self.drawing_params['hist']:
                self.st.list_funcs(kwargs['func'],chi=kwargs['chi'],NDF=kwargs['NDF'])
                self.st.draw()
                self.canv.Update()
        else:
            self.drawing_params['stats'] = False
            del self.st
            self.canv.Update()

    #CMD:"i"
    def __integrate(self):
        if len(self.q) > 0:
            intglo = float(self.q.pop())
            intghi = float(self.q.pop())
        else:
            intglo = float(raw_input("i - integrate from: "))
            intghi = float(raw_input("i - integrate to: "))
        print "i - integral of func on [%4.4f,%4.4f]: %f"%(intglo,intghi,self.func.Integral(intglo,intghi))
        print "i - integral of func on range [%4.4f,%4.4f]: %f"%(self.ftr.lo,self.ftr.hi,self.func.Integral(self.ftr.lo,self.ftr.hi))
        # print "i - integral of func on range [%4.4f,%4.4f]: %f"%(ftr.lo,ftr.hi,func.GetHistogram().Integral())

    #CMD:"c"
    def __cum(self):
        self.canv.Clear()
        if self.drawing_params['hist']:
            self.hist.SetMaximum(1.05*max(self.hist.Integral(),
                    1. if not (self.drawing_params['func'] or self.drawing_params['fhist']) else self.func.GetHistogram().Integral()))
            self.hist.GetCumulative().Draw()
            self.canv.Update()
            print "c - hist Integral",self.hist.Integral()
        if self.drawing_params['fhist']:
            self.func.GetHistogram().GetCumulative().Draw("SAME" if self.drawing_params['hist'] else "")
            self.canv.Update()
            print "c - func histogram Integral",self.func.GetHistogram().Integral()
        if self.drawing_params['func']:
            cumfunc = self.func.Clone()
            cumfunc.SetParameter(0,cumfunc.GetParameter(0)/cumfunc.GetXaxis().GetBinWidth(0))
            cum = ROOT.TGraph(cumfunc.DrawIntegral("SAME" if self.drawing_params['hist'] or self.drawing_params['fhist'] else ""))
            self.canv.Update()
            print "c - func Integral",cumfunc.Integral(self.ftr.lo,self.ftr.hi)
            cumfunc.Print()
        self.drawing_params['cum'] = True

    #CMD:"n"
    def __norm(self):
        if not self.drawing_params['norm']:
            print "n - [NORMALIZING]"
            hscale_factor = 1. / self.hist.Integral()
            # func.SetNpx(hist.GetNbinsX())
            norm_factor = 1. / self.func.GetHistogram().Integral()
            self.func.SetParameter(0,self.func.GetParameter(0)*norm_factor)
            if self.drawing_params['hist']:
                self.hist.GetYaxis().SetTitle("Event Probability Density")
                self.hist = ROOT.TH1D(self.hist.DrawNormalized("HIST"))
                self.canv.Update()
                if self.drawing_params['func'] or self.drawing_params['fhist']:
                    y_ax_max = float(1.05*max([self.func.GetMaximum(),self.hist.GetMaximum()]))
                    self.hist.SetMaximum(y_ax_max)
                    self.func.SetMaximum(y_ax_max)
                    self.canv.Update()
                    # hist.Draw("HIST")
            if self.drawing_params['fhist']:
                self.func.GetHistogram().DrawNormalized("HIST"+("SAME" if self.drawing_params['hist'] else ""))
                self.canv.Update()
            if self.drawing_params['func']:
                self.func.DrawCopy("C"+("SAME" if self.drawing_params['fhist'] or self.drawing_params['hist'] else ""))
                self.canv.Update()
            self.drawing_params['norm'] = True
        if self.drawing_params['legend']:
            if self.drawing_params['hist']:
                self.lgn.list_hists(hist=self.hist,run_file=self.ftr.run_file)
                self.lgn.Draw()
                self.canv.Update()
        if self.drawing_params['stats']:
            if self.drawing_params['func']:
                self.st.list_funcs(func=self.func,chi=self.ftr.chi,NDF=self.ftr.NDF)
                self.st.Draw()
                self.canv.Update()
        
        for n in range(self.func.GetNpar()):
            print self.func.GetParName(n),self.func.GetParameter(n)
        self.func.Print()
        print self.func
        print "n - function normalization factor:",norm_factor
        print "n - histogram scale factor:",hscale_factor
        self.canv.Update()

    #CMD:"h"
    def __hist(self):
        if self.drawing_params['hist']:
            self.hist.Delete()
            chain.Delete()
        self.drawing_params['hist'] = True
        

        x_ax = self.hist.GetXaxis()
        if self.ftr.pname == 'phi':
            x_ax.SetTitle("%s (MeV)"%(self.ftr.var))
        else:
            x_ax.SetTitle("%s (GeV)"%(self.ftr.var))
        x_ax.CenterTitle(True)
        y_ax = self.hist.GetYaxis()
        y_ax.SetTitle("Event Probability Density" if self.drawing_params['norm'] else "Events")
        y_ax.CenterTitle(True)

        self.hist.SetFillColor(kAzure-8)
        self.hist.SetLineColor(kAzure-7)
        self.hist.SetLineWidth(3)
        self.hist.SetStats(0)

        print self.ftr.fit_info
        print self.ftr.run_file

        chain = ROOT.TChain("twoprongNtuplizer/fTree")
        chain.Add(self.ftr.run_file)
        draw_s = self.ftr.var + ">>hist"
        cut_s = self.ftr.cuts
        chain.Draw(draw_s, cut_s,"HIST")
        #y_ax.SetLimits(0.,max(func.GetMaximum(),hist.GetMaximum())*1.05)
        #y_ax.SetMax(500)
        if self.drawing_params['func'] or self.drawing_params['fhist']:
            self.hist.SetMaximum(float(1.05*max([self.func.GetMaximum(),self.hist.GetMaximum()])))
        self.canv.Update()

        if self.drawing_params['fhist']:
            self.func.Draw("HIST SAME")
            self.canv.Update()
        if self.drawing_params['func']:
            self.func.Draw("C SAME")
            self.canv.Update()
        if self.drawing_params['legend']:
            self.lgn.list_hists(self.hist,self.ftr.run_file)
            self.lgn.Draw()
            self.canv.Update()
        if self.drawing_params['stats']:
            self.st.Draw()
            self.canv.Update()

        print "h - hist intg",self.hist.Integral()
        print "h - fhist intg",self.func.GetHistogram().Integral()

    #CMD:"line"
    def __line(self):
        par_names = list(self.func.GetParName(i) for i in range(self.func.GetNpar()))
        par_values = list(self.func.GetParameter(i) for i in range(self.func.GetNpar()))
        pars = zip(par_names,par_values)
        print pars
        if 'Mean' in par_names:
            mean = float(pars[par_names.index('Mean')][1])
        elif 'MPV' in par_names:
            mean = float(pars[par_names.index('MPV')][1])
        else:
            mean = 0.
        val = self.func.Eval(mean)
        cm = self.func.CentralMoment(2,self.ftr.lo,self.ftr.hi)

        vline = ROOT.TLine(mean,0.,mean,self.func.GetMaximum())
        vline.SetLineColor(kGreen)
        vline.SetLineWidth(4)
        vline.Draw("SAME")
        hline1 = ROOT.TLine(self.ftr.lo,val,mean,val)
        hline1.SetLineColor(kMagenta)
        hline1.SetLineWidth(4)
        hline1.Draw("SAME")
        hline2 = ROOT.TLine(cm,0.,cm,self.func.GetMaximum())
        hline2.SetLineColor(kCyan)
        hline2.SetLineWidth(4)
        hline2.Draw("SAME")
        self.canv.Update()
        
        print "line - mean:",mean
        print "line - value at mean:",val
        print "line - central moment",cm

    #CMD:"p"
    def __pars(self):
        par_names = list(self.func.GetParName(i) for i in range(self.func.GetNpar()))
        par_values = list(self.func.GetParameter(i) for i in range(self.func.GetNpar()))
        pars = zip(par_names,par_values)
        print "p - ",pars
        print "p - Chi-Squared",self.ftr.chi

    #CMD:"json"
    def __json_dump(self):
        print "json-p - File:",self.fit_info
        with open(self.fit_info) as json_file:
            info = json.load(json_file)
            print json.dumps(info,indent=4)

    #CMD:"+"
    def __add(self):
        fit_info2 = "${PROJECT_DIR}/"+raw_input("+ - new fit init file to add: ${PROJECT_DIR}/")
        ftr2 = fitter(fit_info2)
        ftr2.func.SetNpx(ftr2.bins)
        func2 = ftr2.func.Clone()
        func2.SetLineColor(kBlue-3)
        func2.SetLineWidth(5)
        
        func.Draw()
        func2.Draw("same")

        print "+ - 1st func X^2",self.ftr.chi
        print "+ - 2nd func X^2",ftr2.chi

        par_names = list(self.func.GetParName(i) for i in range(self.func.GetNpar()))
        par_values = list(self.func.GetParameter(i) for i in range(self.func.GetNpar()))
        par_values2 = list(self.func2.GetParameter(i) for i in range(self.func.GetNpar()))
        par_avgs = list((par_values2[i] + par_values[i])/2 for i in range(self.func.GetNpar()))
        par_diffs = list(par_values2[i] - par_values[i] for i in range(self.func.GetNpar()))
        par_pcts = list(abs(par_diffs[i])/par_avgs[i] for i in range(self.func.GetNpar()))
        pars = zip(par_names,par_avgs,par_pcts)
        
        print "+ - parameter comparison\n",pars
        self.canv.Update()

    #CMD:"*"
    def __new(self):
        if len(self.q) > 0:
            affirm = self.q.pop()
        else:
            affirm = raw_input("* - are you sure you want to RESTART with a new file? [y/n]")
        if affirm == 'y':
            del self.lgn
            self.canv.Clear()
            fit_info = "${PROJECT_DIR}/"+raw_input("* - name of fit info .json: ${PROJECT_DIR}/")
            print "USING FIT INFO",fit_info
            self.ftr = fitter(fit_info)
            self.ftr.func.SetNpx(self.ftr.bins)
            func = ftr.func.Clone()
            func.SetLineColor(kPink)
            func.SetLineWidth(5)
            for e in self.drawing_params:
                self.drawing_params[e] = False
            q = deque()
            self.canv.Update()

    #CMD:"/"
    def __clear(self):
        self.func = self.ftr.func.Clone()
        self.func.SetLineColor(kPink)
        self.func.SetLineWidth(5)
        if 'lgn' in locals():
            lgn.Delete()
        for e in self.drawing_params:
            self.drawing_params[e] = False
        self.q = deque()
        self.canv.Clear()
        self.canv.Update()

    #CMD:"cmd-q"
    def __fill_q(self):
        qstring = raw_input("cmd-q - input string of commands separated by ',' (1 line): ")
        qlist = qstring.split(',')
        qlist.reverse()
        self.q = deque(qlist)

if __name__ == "__main__":
    print sys.argv
    print sys.path