import ROOT
import time
import json
import re
import os
from datetime import datetime

from ROOT import Math
from ROOT import TMath

from ROOT import kAzure
from ROOT import kRed
from ROOT import kPink

from defs import PROJECT_DIR
from defs import file_path_check as ck

class fitter:

    fits = {
        'gaus': 'gaus',
        'crystalball': 'crystalball',
        'landau': 'landau',
        'landxgaus': 'CONV(landau,gaus)',
        'landrefl': '[0]*TMath::Landau(-x,-[1],[2])',
        'gausgaus': 'NSUM(gaus,gaus)'
    }

    class FitterInitializeException(Exception):
        """
        Exception notifying when a build.fitter instance
        has not been initialized correctly
        """
        def __init__ (self, message):
            self.message = ("builder.fitter is initialized with:"
                            /"\n - 0 args (fit with src/build/fitter-init.json))"
                            /"\n - 1 arg (fit with path to fit information file provided)")

    #create fitter instance (two different methods)
    def __init__(self,*args):
        if len(args) == 0:
            self.__init_new()
        elif len(args) == 1:
            self.__init_fitted(args[0])
        else:
            raise FitterInitializeException()

    #unfitted fitter
    def __init_new(self):
        self.fit_info = PROJECT_DIR+"/src/build/fitter-init.json"
        with open(self.fit_info,'r') as json_file:
            info = json.load(json_file)
        
        run_info = info['run_info']
        self.run_file = run_info['run_file']
        self.run_name = run_info['run_name']
        self.phi = run_info['run_parameters']['phi']
        self.omega = run_info['run_parameters']['omega']

        self.run_file = ck(info['run_info']['run_file'])
        self.fit_name = info['fit_name']

        self.cuts = info['cuts']
        self.particle = info['particle']
        self.mean_estimate = run_info['run_parameters'][self.particle]
        particle_info = info[self.particle]
        self.var = particle_info['var'] if not (self.particle == 'omega' and self.omega == 0.950) else particle_info['var'][:-6]+"Eta[0]"
        s = particle_info['bins'].split(',')
        self.bins = int(s[0])
        self.lo = float(s[1])
        self.hi = float(s[2])

        self.debug = int(info['debug'])
        self.fix_constant = info['fix_constant']
        self.func = None
        self.hist = None
        self.chi = None
        self.NDF = None
        self.chiprob = None

    #fitted fitter
    def __init_fitted(self,fit_info):
        self.fit_info = ck(fit_info)
        with open(self.fit_info,'r') as json_file:
            info = json.load(json_file)
        
        self.run_file = ck(info['run_info']['run_file'])
        self.fit_name = info['fit_name']

        self.cuts = info['cuts']
        self.particle = info['particle']
        self.fit_name = info['fit_name']
        self.var = info['var']
        s = info['bins'].split(',')
        self.bins = int(s[0])
        self.lo = float(s[1])
        self.hi = float(s[2])
        self.func = ROOT.TF1(self.fit_name, self.fits[self.fit_name], self.lo, self.hi)
        self.func.SetNpx(self.bins)
        for i in range(self.func.GetNpar()):
            self.func.SetParameter(i, info['pars']['%i' % (i)])
            self.func.SetParName(i, info['names']['%i' % (i)])
        self.chi = info['chi']
        self.NDF = info['NDF']
        self.chiprob = info['chiprob']
        self.fix_constant = info['fix_constant']

    def __str__(self):
        with open(self.fit_info) as json_file:
            info = json.load(json_file)
            return str(info)

    def __repr__(self):
        with open(self.fit_info) as json_file:
            info = json.load(json_file)
            return repr(info)

    def fit(self):

        run_file = self.run_file
        fit_name = self.fit_name
        debug = self.debug

        # prepare the plot, axes, and name of histogram for display
        self.hist = self.__setup_hist()

        # import data from .root file to fit and display on hist
        self.__import_data(run_file,self.var,self.cuts,debug=debug)

        # threshold for too-low events - focus on peak, not tails
        # self.__adjust_hist()

        # get TFormula expression for given fit keyword from available fits
        if fit_name in self.fits.keys():
            fit_formula = self.fits[fit_name]
            print "MODELLING %s %.3f GEV WITH %s" % (self.particle,self.mean_estimate,fit_formula)
            func = ROOT.TF1("func", fit_formula, self.lo, self.hi)
            func.SetNpx(self.bins)
        else:
            print "FIT NOT COMPATIBLE...exiting"
            exit

        # initial seed of func parameters with gaussian parameters
        # before fitting each parameter individually
        gaus_func = self.__gaus_seed(self.hist.Clone(),self.mean_estimate)
        fit_func = self.__fit_all_parameters(gaus_func,func,self.hist,fit_name)

        self.chi = fit_func.GetChisquare()
        self.NDF = fit_func.GetNDF()
        self.chiprob = fit_func.GetProb()
        self.func = fit_func

    # create histogram to be filled with data from input .root file
    def __setup_hist(self):
        hist = ROOT.TH1D("hist", "%s fit - %s, phi=%.0f, omega=%.3f" %
                        (self.fit_name,self.run_name,self.phi,self.omega), self.bins, self.lo, self.hi)
        hist.GetXaxis().SetTitle("Mass (GeV)")
        hist.GetXaxis().CenterTitle(True)
        hist.GetYaxis().SetTitle("Events")
        hist.GetYaxis().CenterTitle(True)
        hist.SetStats(0)
        hist.SetFillColor(kAzure-8)
        hist.SetLineColor(kAzure-7)
        hist.SetLineWidth(3)
        return hist

    # import data from .root file for fitting and drawing. 
    # requires __setup_hist() to be run and/or instance of TH1D 'hist'
    def __import_data(self,run_file,var,cuts,debug=0):
        chain = ROOT.TChain("twoprongNtuplizer/fTree")
        chain.Add(run_file)
        draw_string = var + ">>hist"
        cut_string = cuts
        if debug >= 2:
            print "DRAWING ROOT DATA HISTOGRAM"
        chain.Draw(draw_string, cut_string, "goff" if debug <= 1 else "")

    # cut out extraneous/low-event bins
    # function currently NOT IN USE
    def __adjust_hist(self):
        # cut out small bins
        for bin in range(self.hist.GetNbinsX()):
            if self.hist.GetBinContent(bin) < self.thresh:
                self.hist.SetBinContent(bin,0.)

    # need better seeding methods

    def __gaus_seed(self,hist,mean_estimate):
        gaus_func = ROOT.TF1("gaus_seed","gaus",self.lo,self.hi)
        gaus_func.SetNpx(self.bins)
        gaus_func.SetLineWidth(5)
        hist.Fit(gaus_func,"SM0Q")
        gaus_func = hist.GetFunction("gaus_seed").Clone()
        print "gaus chi-square",gaus_func.GetChisquare()
        return gaus_func
    
    def __fit_all_parameters(self,gaus_func,func,hist,fit_name):
        
        gaus_const = gaus_func.GetParameter("Constant")
        gaus_mean = gaus_func.GetParameter("Mean")
        gaus_width = gaus_func.GetParameter("Sigma")

        ctemp = ROOT.TCanvas("ctemp","Temp Fitter Plots",1200,900)
        print "GAUS PRE-FIT:\nConst - %.4f\nMean - %4.4f\nWidth - %3.4f"%(
            gaus_const,gaus_mean,gaus_width)

        if fit_name in ['gaus','crystalball','landau','landxgaus','landrefl'] :
            func.SetParameter(0,gaus_const)
            func.SetParameter(1,gaus_mean)
            func.SetParameter(2,gaus_width)
        if fit_name == "crystalball":
            func.SetParameter("Alpha",5)
            func.SetParameter("N",5)
        if fit_name == 'landxgaus':
            func.SetParameter(3,gaus_mean)
            func.SetParameter(4,gaus_width)
        if fit_name == 'gausgaus':
            func.SetParameters(gaus_const,gaus_const,
                gaus_mean,gaus_mean,gaus_width,gaus_width)
        
        fit_options = "M0BQ"
        if self.debug == 0:
            fit_options += "Q"
        elif self.debug == 3:
            fit_options += "V"

        '''
        S for Save function in histogram fit
        M for search for more minima after initial fit
        0 for don't plot
        L for NLL minimization method instead of X^2
        Q for Quiet
        V for Verbose
        R for Range defined in TF1 def
        B for fixing parameters to those defined in fn pre-fit
        '''

        n = func.GetNpar()

        for i in range(n):
            print func.GetParName(i),func.GetParameter(i)
        hist.Fit("func",fit_options)
        print "chi-square",func.GetChisquare()

        # fit the function to the histogram while sequentially fixing each parameter
        # always refit the constant for best results
        for i in range(1 if fit_name != 'gausgaus' else 2,n):
            # start fitting by fixing constant first to hold the peak of the distribution
            if self.fix_constant:
                func_max = func.GetMaximum(self.lo,self.hi)
                hist_max = hist.GetBinContent(hist.GetMaximumBin())
                print "func_max:",func_max,"\thist_max",hist_max
                scale = hist_max/func_max
                print "scale",scale
                func.FixParameter(0,scale*func.GetParameter(0))
            
            # if no fixing constant, start fitting by fixing mean first, constant last
            p = func.GetParameter(i)
            func.FixParameter(i,p)
            print "%s fit - fix parameter#%i = %4.4f"%(fit_name,i,p)
            hist.Fit("func",fit_options)
            print "chi-square",func.GetChisquare()

        print "final fit after unfixing parameters"
        
        print "before unfixed fit"
        for i in range(n):
            # if self.fix_constant and i == 0:
            #     continue
            func.ReleaseParameter(i)
            print "releasing parameter",func.GetParName(i),func.GetParameter(i)
        hist.Fit("func",fit_options+"S")
        print "chi-square",func.GetChisquare()
        
        print "after unfixed fit"
        for i in range(n):
            print func.GetParName(i),func.GetParameter(i)
        
        return func

    def jsonify(self,fit_info=""):
        if fit_info == "":

            particle_dir = PROJECT_DIR+"/out/"+self.particle
            fit_name_dir = particle_dir+"/"+self.fit_name
            #particle directory
            if not os.path.isdir(particle_dir):
                os.mkdir(particle_dir)
            #fit_name directory
            if not os.path.isdir(fit_name_dir):
                os.mkdir(fit_name_dir)
            
            fit_info = fit_name_dir+"/%s_%s_PH-%04i_OM-%sp%s.json" % (
                self.run_name,self.fit_name,self.phi,str(self.omega)[0:1],str("%04i"%(int(self.omega*1000)))[1:4])
            
            print "SAVED ftr DATA TO JSON:",fit_info
        
        info = {}
        
        info['fit_name'] = self.fit_name
        info['particle'] = self.particle
        info['cuts'] = self.cuts
        info['var'] = self.var
        info['bins'] = "%i,%f,%f"%(self.bins,self.lo,self.hi)
        pars = {}
        names = {}

        for i in range(self.func.GetNpar()):
            pars.update({i: self.func.GetParameter(i)})
            names.update({i: self.func.GetParName(i)})

        info['pars'] = pars
        info['names'] = names
        info['chi'] = self.chi
        info['NDF'] = self.NDF
        info['chiprob'] = self.chiprob

        run_info = {}
        run_info['run_file'] = re.sub(PROJECT_DIR,r"${PROJECT_DIR}",self.run_file)
        run_info['run_name'] = self.run_name
        run_info['phi_mass'] = self.phi
        run_info['omega_mass'] = self.omega

        info['run_info'] = run_info

        info['fix_constant'] = self.fix_constant
        
        with open(fit_info, 'w') as json_out:
            json.dump(info, json_out, indent=4)

        return fit_info
