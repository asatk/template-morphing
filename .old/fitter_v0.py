import sys
import os
import ROOT
import math
import string
import time
import random
import json
import re
from datetime import datetime

from ROOT import Math
from ROOT import TMath

class fitter:

    fits = {
        'gaus':'gaus',
        'gaus_pdf':'[0]*ROOT::Math::normal_pdf(x[0],[2],[1])',
        'gaus_cdf':'[0]*ROOT::Math::normal_cdf(x[0],[2],[1])',
        'dbl_gaus':'gaus(0)+gaus(3)',
        'dbl_gaus_cdf':'',
        'crys_ball':'crystalball',
        'crys_ball_fn':'[0]*ROOT::Math::crystalball_function(x[0],[3],[4],[2],[1])',
        'crys_ball_pdf':'[0]*ROOT::Math::crystalball_pdf(x[0],[3],[4],[2],[1])',
        'crys_ball_cdf':'[0]*ROOT::Math::crystalball_cdf(x[0],[3],[4],[2],[1])',
        'landau':'landau',
        'landau_pdf':'[0]*ROOT::Math::landau_pdf(x[0],[1],[2])',
        'landau_cdf':'[0]*ROOT::Math::landau_cdf(x[0],[1],[2])',
        'landxgaus':'',
        'landxgaus_cdf':''
    }

    def __init__(self,fit_name='gaus',fit_info='fitter-init.json'):
        with open(fit_info) as json_file:
            info = json.load(json_file)
            self.fileName = info['fname']
            self.fit_name = fit_name
            self.pname = info['pname']
            p_info = info[self.pname][0]
            self.var = p_info['var']
            s = p_info['bins'].split(',')
            self.bins = int(s[0])
            self.lo = float(s[1])
            self.hi = float(s[2])
            self.cuts = info['cuts']
            self.user_data = info['user_data'] == 'y'
            self.normalize = info['normalize'] == 'y'
            self.cum = info['cum'] == 'y'
            self.nEntries = int(info['nEntries'])
            self.gaus_seed = info['seed'] == 'gaus'
            self.debug = int(info['debug'])
            self.window_time = int(info['window_time'])
            self.seed_func = None
            self.func = None
            self.hist = None
            self.cum_func = None

    def __str__(self):
        with open('fitter-init.json') as json_file:
            info = json.load(json_file)
            return str(info)
    def __repr__(self):
        with open('fitter-init.json') as json_file:
            info = json.load(json_file)
            return repr(info)

    def get_fit(self):

        fileName = self.fileName
        fit_name = self.fit_name
        debug = self.debug

        random.seed(datetime.now())

        if debug >= 1:
            print "TESTING %s FIT"%(fit_name)
        hist = ROOT.TH1D("hist","plot %s fit"%(fit_name),self.bins,self.lo,self.hi)
        hist.SetFillColor(45)

        mean_est = 0
        omega_scaling = 1 #scale start parameters of fit by 1/100 for omega analysis
        if self.user_data:
            chain = ROOT.TChain("twoprongNtuplizer/fTree")
            chain.Add(fileName)
            draw_s = self.var + ">>hist"
            cut_s = self.cuts
            if debug == 2:
                print "DRAWING ROOT DATA HISTOGRAM"
            chain.Draw(draw_s,cut_s,"goff" if debug <= 1 else "")
            self.nEntries = hist.Integral()
            if self.pname == 'phi':
                pmass = re.search("(\d+)(?=\.root)",fileName).group()
                mean_est = float(pmass)
            elif self.pname == 'omega':
                pprime = bool(re.search("prime",fileName) is not None)
                mean_est = 0.95 if pprime else 0.55
                omega_scaling = 1/100
            else:
                pass
            if debug >= 1:
                    print "RECO MASS MODELLING FOR:",self.pname,mean_est
        else:
            hist.FillRandom(fit_str,nEntries)
            if debug == 2: 
                print "DRAWING RANDOM FILL HISTOGRAM"
                hist.Draw()

        fit_str = ""
        if fit_name in self.fits.keys():
            fit_str = self.fits[fit_name]
            if debug >= 1: 
                print "MODELLING WITH %s"%(fit_str)
            fn = ROOT.TF1("fit",fit_str,self.lo,self.hi)
        else:
            if debug >= 1: 
                print "FIT NOT COMPATIBLE...exiting"
            exit

        if fit_name in ['gaus','gaus_pdf','gaus_cdf']:
            fn.SetParName(0,'Constant')
            fn.SetParName(1,'Mean')
            fn.SetParName(2,'Sigma')
            fn.SetParameter('Constant',random.uniform(1./30,1./3) if self.normalize else self.nEntries*random.uniform(1./30,1./3))
            fn.SetParameter('Mean',mean_est+mean_est*random.uniform(-0.05,0.05))
            fn.SetParameter('Sigma',random.uniform(0.001,0.25) if self.pname == 'omega' else random.randint(0,25))
        elif fit_name in ['dbl_gaus','dbl_gaus_cdf']:
            fn.SetParName(0,'Constant-1')
            fn.SetParName(1,'Mean-1')
            fn.SetParName(2,'Sigma-1')
            fn.SetParName(3,'Constant-2')
            fn.SetParName(4,'Mean-2')
            fn.SetParName(5,'Sigma-2')
            fn.SetParameters(random.uniform(1./30,1./3) if self.normalize else self.nEntries*random.uniform(1./30,1./3),mean_est+mean_est*random.uniform(-0.05,0.05),random.randint(0,100),\
                random.uniform(1./30,1./3) if self.normalize else self.nEntries*random.uniform(1./30,1./3),mean_est+mean_est*random.uniform(-0.05,0.05),random.randint(0,100))
        elif fit_name in ['crys_ball','crys_ball_fn','crys_ball_pdf','crys_ball_cdf']:
            fn.SetParName(0,'Constant')
            fn.SetParName(1,'Mean')
            fn.SetParName(2,'Sigma')
            fn.SetParName(3,'Alpha')
            fn.SetParName(4,'n')
            fn.SetParameter('Constant',random.uniform(1./30,1./3) if self.normalize else self.nEntries*random.uniform(1./30,1./3))
            fn.SetParameter('Mean',mean_est+mean_est*random.uniform(-0.05,0.05))
            fn.SetParameter('Sigma',random.uniform(0.001,0.25) if self.pname == 'omega' else random.randint(0,25))
            fn.SetParameter('Alpha',random.random()*25)
            fn.SetParameter('n',random.randint(1,10))
        elif fit_name in ['landau','landau_pdf','landau_cdf']:
            fn.SetParName(0,'Constant')
            fn.SetParName(1,'eta/Width')
            fn.SetParName(2,'x0/Mean')
            fn.SetParameter('Constant',random.uniform(1./30,1./3) if self.normalize else self.nEntries*random.uniform(1./30,1./3))
            fn.SetParameter('eta/Width',random.uniform(0.001,0.1) if self.pname == 'omega' else random.randint(0,100))
            fn.SetParameter('x0/Mean',mean_est+mean_est*random.uniform(-0.05,0.05))
        elif fit_name in ['landxgaus','landxgaus_cdf']:
            fn_conv = ROOT.TF1Convolution('landau','gaus',self.lo,self.hi,True)
            fn_conv.SetRange(self.lo,self.hi)
            fn_conv.SetNofPointsFFT(1000)
            fn = ROOT.TF1('fit',fn_conv,self.lo,self.hi,fn_conv.GetNpar())
            for i in range(fn_conv.GetNpar()):
                fn.SetParameter(i,random.random()*random.randint(0,mean_est))

        if debug == 2: 
            print "SEED PARAMETERS:"
            for i in range(fn.GetNpar()):
                print "PAR %i:\t %s \t| %s"%(i,fn.GetParName(i)[:4],fn.GetParameter(i))
        
        if self.normalize:
            hist.normalize(1/hist.Integral())
        if self.cum:
            hist = hist.GetCumulative()
            if debug >= 2:
                print "DRAWING CUM HISTOGRAM"
                hist.Draw()

        fit_opts = "SM0"
        if debug == 0:
            fit_opts += "Q"
        elif debug == 3:
            fit_opts += "V"
        fit_res = hist.Fit("fit",fit_opts)
        #S for Save function in histogram fit
        #M for search for more minima after initial fit
        #0 for don't plot
        #L for NLL minimization method instead of X^2
        #Q for Quiet
        #V for Verbose
        #R for Range defined in TF1 def
        #B for fixing parameters to those defined in fn pre-fit

        fit_fn = hist.GetFunction("fit")
        self.func = hist.GetFunction("fit")
        self.hist = hist

        fit_fn.SetLineColor(4)
        
        if debug == 2:
            print "DRAWING FIT"
            fit_fn.Draw("same")
            print "VALUE AT THEORETICAL MEAN ",fn.Eval(mean_est)
            print "FIT PARAMETERS FOR %s"%(fit_str)
            for i in range(fit_fn.GetNpar()):
                print "PAR %i:\t %s \t| %s"%(i,fit_fn.GetParName(i)[:4],fit_fn.GetParameter(i))
            print "CHI SQUARED FOR FIT ",fit_fn.GetChisquare()
            time.sleep(self.window_time)

        if debug >= 1:
            print "TEST %s - END"%(fit_name)
        
        return self.func

    def get_cum(self):
        self.cum_func = ROOT.TF1('cum_func',self.__cum__,self.lo,self.hi,1)
        print self.cum_func
        if self.debug == 2:
            self.cum_func.Draw()
        return self.cum_func
    
    def __cum__(self,x,par):
        return par[0]*self.func.Integral(self.lo,x[0])

    def jsonify(self):
        pass