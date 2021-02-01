import ROOT
import time
import random
import json
import re
import os
from datetime import datetime

from ROOT import Math
from ROOT import TMath

from ROOT import kAzure
from ROOT import kRed
from ROOT import kPink

class fitter:

    fits = {
        'gaus': 'gaus',
        'gaus_cdf': '[0]*ROOT::Math::normal_cdf(x[0],[2],[1])',
        'crystalball': 'crystalball',
        'crystalball_fn': '[0]*ROOT::Math::crystalball_function(x[0],[3],[4],[2],[1])',
        'crystalball_cdf': '[0]*ROOT::Math::crystalball_cdf(x[0],[3],[4],[2],[1])',
        'landau': 'landau',
        'landau_cdf': '[0]*ROOT::Math::landau_cdf(x[0],[2],[1])',
        'landxgaus': 'CONV(landau,gaus)',
        'landxgaus_cdf': ''
    }

    class FitterInitializeException(Exception):
        """
        Exception notifying when a build.fitter instance
        has not been initialized correctly
        """
        def __init__ (self, message):
            self.message = ("builder.fitter initialized with:"+
                            "\n - 0 args (fit with src/build/fitter-init.json))"+
                            "\n - 1 arg (fit with path to fit information file provided)")

    #unfitted fitter
    def __init__(self,*args):
        if len(args) == 0:
            self.__init_new()
        elif len(args) == 1:
            self.__init_fitted(args[0])
        else:
            raise FitterInitializeException()

    #unfitted fitter
    def __init_new(self):
        self.fit_info = "./build/fitter-init.json"
        with open(self.fit_info,'r') as json_file:
            info = json.load(json_file)
        
        self.file_name = info['file_name']
        self.fit_name = info['fit_name']

        self.cuts = info['cuts']
        self.pname = info['pname']
        p_info = info[self.pname][0]
        self.var = p_info['var'] if not 'prime' in self.file_name or self.pname != 'omega' else p_info['var'][:-6]+"Eta[0]"
        s = p_info['bins'].split(',')
        self.bins = int(s[0])
        self.lo = float(s[1])
        self.hi = float(s[2])
        self.normalized = info['normalized']
        self.nEntries = int(info['nEntries'])
        self.seed = info['seed']
        self.debug = int(info['debug'])
        self.window_time = int(info['window_time'])
        self.func = None
        self.hist = None
        self.chi = None
        self.NDF = None
        self.chiprob = None

    #fitted fitter
    def __init_fitted(self,fit_info):
        self.fit_info = fit_info
        with open(self.fit_info,'r') as json_file:
            info = json.load(json_file)
        
        self.file_name = info['file_name']
        self.fit_name = info['fit_name']

        self.cuts = info['cuts']
        self.pname = info['pname']
        self.fit_name = info['fit_name']
        self.var = info['var']
        s = info['bins'].split(',')
        self.bins = int(s[0])
        self.lo = float(s[1])
        self.hi = float(s[2])
        self.func = ROOT.TF1(self.fit_name, self.fits[self.fit_name], self.lo, self.hi)
        self.func.SetNpx(self.bins)
        self.normalized = info['normalized']
        for i in range(self.func.GetNpar()):
            self.func.SetParameter(i, info['pars']['%i' % (i)])
            self.func.SetParName(i, info['names']['%i' % (i)])
        self.chi = info['chi']
        self.NDF = info['NDF']
        self.chiprob = info['chiprob']

    def __str__(self):
        with open(self.fit_info) as json_file:
            info = json.load(json_file)
            return str(info)

    def __repr__(self):
        with open(self.fit_info) as json_file:
            info = json.load(json_file)
            return repr(info)

    def fit(self):

        file_name = self.file_name
        fit_name = self.fit_name
        debug = self.debug

        random.seed(datetime.now())

        if debug >= 1:
            print "TESTING %s FIT\n" % (fit_name)
        hist = ROOT.TH1D("hist", "plot %s fit" %
                         (fit_name), self.bins, self.lo, self.hi)
        hist.GetXaxis().SetTitle("Mass (GeV)")
        hist.GetXaxis().CenterTitle(True)
        if self.normalized:
            hist.GetYaxis().SetTitle("Normalized Distribution")
        else:
            hist.GetYaxis().SetTitle("Events")
        hist.GetYaxis().CenterTitle(True)
        hist.SetStats(0)
        hist.SetFillColor(kAzure-8)
        hist.SetLineColor(kAzure-7)
        hist.SetLineWidth(3)

        mean_est = 0
        omega_scaling = 1.  # scale start parameters of fit by 1/100 for omega analysis,
        # try to remove this par by using better/more general seeds

        # get data from .ROOT file, store in histogram
        chain = ROOT.TChain("twoprongNtuplizer/fTree")
        chain.Add(file_name)
        draw_s = self.var + ">>hist"
        cut_s = self.cuts
        if debug >= 2:
            print "DRAWING ROOT DATA HISTOGRAM"
        chain.Draw(draw_s, cut_s, "goff" if debug <= 1 else "")
        self.nEntries = hist.Integral()
        # get reco mass estimate for particle
        if self.pname == 'phi':
            pmass = re.search("(\d+)(?=\.root)", file_name).group()
            mean_est = float(pmass)
        elif self.pname == 'omega':
            pprime = bool(re.search("prime", file_name) is not None)
            mean_est = 0.95 if pprime else 0.55
            omega_scaling = 1. / 1000
        if self.normalized:
            hist = ROOT.TH1D(hist.DrawNormalized("HIST"))
        if debug >= 1:
            print "RECO MASS MODELLING FOR:", self.pname, mean_est

        # get TFormula expression for given fit keyword from available fits
        fit_str = ""
        if fit_name in self.fits.keys():
            fit_str = self.fits[fit_name]
            if debug >= 1:
                print "MODELLING WITH %s" % (fit_str)
            fn = ROOT.TF1("fit", fit_str, self.lo, self.hi)
            fn.SetNpx(self.bins)
        else:
            if debug >= 1:
                print "FIT NOT COMPATIBLE...exiting"
            exit

        if self.seed == 'file':
            # format file name to appropriate convention
            seed_file_str = "../out/fits/%sfitter-%s-%s-%s.json"
            eta_start = self.file_name.find('eta')
            num_match = re.search("\\d+(?=\\.root)",self.file_name[eta_start:])
            num = int(num_match.group())
            num_start = int(num_match.start())
            seed_file = seed_file_str%(
                    "norm" if self.normalized else "",
                    self.fit_name,self.pname,
                    (self.file_name[eta_start:eta_start + num_start] + "%04i"%(num)))
            print "USING SEED: " + seed_file
        # set up and create random seeds for fn parameters
        if fit_name in ['gaus', 'gaus_cdf']:
            fn.SetParName(0, 'Constant')
            fn.SetParName(1, 'Mean')
            fn.SetParName(2, 'Sigma')
            if self.seed == 'rand':
                fn.SetParameter('Constant', random.uniform(
                        1. / 30, 1. / 3) if self.normalized else self.nEntries * random.uniform(1. / 30, 1. / 3))
                fn.SetParameter('Mean', mean_est + mean_est *
                        random.uniform(-0.05, 0.05))
                fn.SetParameter('Sigma', random.randint(1, 25) * omega_scaling)
                #fn.SetParLimits(0, 0., self.nEntries)
                fn.SetParLimits(1, 0., mean_est * 2)
            elif self.seed == 'file':
                with open(seed_file) as json_file:
                    seed_info = json.load(json_file)
                    for i in range(fn.GetNpar()):
                        fn.SetParameter(i, seed_info['pars']['%i' % (i)])
                        fn.SetParName(i, seed_info['names']['%i' % (i)])
        elif fit_name in ['crystalball', 'crys_ball_fn', 'crys_ball_cdf']:
            fn.SetParName(0, 'Constant')
            fn.SetParName(1, 'Mean')
            fn.SetParName(2, 'Sigma')
            fn.SetParName(3, 'Alpha')
            fn.SetParName(4, 'n')
            if self.seed == 'rand':
                fn.SetParameter('Constant', random.uniform(
                        1. / 30, 1. / 3) if self.normalized else self.nEntries * random.uniform(1. / 30, 1. / 3))
                fn.SetParameter('Mean', mean_est + mean_est *
                        random.uniform(-0.05, 0.05))
                fn.SetParameter('Sigma', random.randint(0, 25) * omega_scaling)
                fn.SetParameter('Alpha', random.random() * 25)
                fn.SetParameter('n', random.randint(1, 10))
                #fn.SetParLimits(0, 0., self.nEntries)
                fn.SetParLimits(1, 0., mean_est * 2)
                fn.SetParLimits(4, 0., 10e6)
            elif self.seed == 'file':
                with open(seed_file) as json_file:
                    seed_info = json.load(json_file)
                    for i in range(fn.GetNpar()):
                        fn.SetParameter(i, seed_info['pars']['%i' % (i)])
                        fn.SetParName(i, seed_info['names']['%i' % (i)])
        elif fit_name in ['landau', 'landau_cdf']:
            fn.SetParName(0, 'Constant')
            fn.SetParName(1, 'MPV')
            fn.SetParName(2, 'Eta')
            if self.seed == 'rand':
                fn.SetParameter('Constant', random.uniform(
                        1. / 30, 1. / 3) if self.normalized else self.nEntries * random.uniform(1. / 30, 1. / 3))
                fn.SetParameter('MPV', mean_est + mean_est *
                        random.uniform(-0.05, 0.05))
                fn.SetParameter('Eta', random.randint(0, 100) * omega_scaling)
                #fn.SetParLimits(0, 0., self.nEntries)
                fn.SetParLimits(1, 0., mean_est * 2)
            elif self.seed == 'file':
                with open(seed_file) as json_file:
                    seed_info = json.load(json_file)
                    for i in range(fn.GetNpar()):
                        fn.SetParameter(i, seed_info['pars']['%i' % (i)])
                        fn.SetParName(i, seed_info['names']['%i' % (i)])
        elif fit_name in ['landxgaus', 'landxgaus_cdf']:
            fn.SetParName(0, 'Constant')
            fn.SetParName(1, 'MPV')
            fn.SetParName(2, 'Eta')
            fn.SetParName(3, 'Mean')
            fn.SetParName(4, 'Sigma')
            if self.seed == 'rand':
                fn.SetParameter('Constant', random.uniform(
                        1. / 30, 1. / 3) if self.normalized else self.nEntries * random.uniform(1. / 30, 1. / 3))
                fn.SetParameter('MPV', mean_est + mean_est *
                        random.uniform(-0.05, 0.05))
                fn.SetParameter('Eta', random.randint(0, 100) * omega_scaling)
                fn.SetParameter('Mean', mean_est + mean_est *
                        random.uniform(-0.05, 0.05))
                fn.SetParameter('Sigma', random.randint(1, 25) * omega_scaling)
                #fn.SetParLimits(0, 0., self.nEntries)
                fn.SetParLimits(1, 0., mean_est * 2)
                fn.SetParLimits(3, 0., mean_est * 2)
            elif self.seed == 'file':
                with open(seed_file) as json_file:
                    seed_info = json.load(json_file)
                    for i in range(fn.GetNpar()):
                        fn.SetParameter(i, seed_info['pars']['%i' % (i)])
                        fn.SetParName(i, seed_info['names']['%i' % (i)])
        
        # print seed parameters for given fit set previously
        if debug >= 2:
            print "SEED PARAMETERS:"
            for i in range(fn.GetNpar()):
                print "PAR %i:\t %s \t| %s" % (i, fn.GetParName(i)[:4], fn.GetParameter(i))
            print "\n"

        # perform fit with options
        fit_opts = "SM0"
        if debug == 0:
            fit_opts += "Q"
        elif debug == 3:
            fit_opts += "V"
        fit_res = hist.Fit("fit", fit_opts)
        fit_fn = hist.GetFunction("fit")

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

        # normalize histogram POST-fit
        if self.normalized:
            # norm_factor = 1. / fit_fn.GetHistogram().Integral()
            # fit_fn.SetParameter(0,fit_fn.GetParameter(0)*norm_factor)
            if debug >= 2:
                #print "\nNORMALIZING HIST BY FACTOR OF %f" % (1. / self.nEntries)
                #print "intg", self.nEntries
                print "NORMALIZING FUNC BY FACTOR OF %f" % (norm_factor)
                print "intg func", 1. / norm_factor
                # fhist.DrawNormalized("same HIST")
                fit_fn.Draw("same C")

        # show plots and fit info during fitting
        if debug >= 2:
            fit_fn.SetLineColor(kPink)
            print "\nDRAWING FIT"
            # hist.Draw("HIST")
            # fit_fn.Draw("same")
            print "VALUE AT THEORETICAL MEAN ", fit_fn.Eval(mean_est)
            print "\nFIT FORMULA"
            #print fit_fn.GetFormula().Print()
            print "\nFIT PARAMETERS FOR %s" % (fit_str)
            for i in range(fit_fn.GetNpar()):
                print "PAR %i:\t %s \t| %s" % (i, fit_fn.GetParName(i)[:4], fit_fn.GetParameter(i))
            time.sleep(self.window_time)

        self.chi = fit_fn.GetChisquare()
        self.NDF = fit_fn.GetNDF()
        self.chiprob = fit_fn.GetProb()

        if debug >= 1:
            print "CHI SQUARED FOR FIT ", self.chi
            print "\nTEST %s - END\n" % (fit_name)

        self.func = fit_fn
        self.hist = hist

        return self.func

    def __gaus_seed(self,):
        fn.SetParName(0, 'Constant')
        fn.SetParName(1, 'Mean')
        fn.SetParName(2, 'Sigma')
        if self.seed == 'rand':
            fn.SetParameter('Constant', random.uniform(
                    1. / 30, 1. / 3) if self.normalized else self.nEntries * random.uniform(1. / 30, 1. / 3))
            fn.SetParameter('Mean', mean_est + mean_est *
                    random.uniform(-0.05, 0.05))
            fn.SetParameter('Sigma', random.randint(1, 25) * omega_scaling)
            #fn.SetParLimits(0, 0., self.nEntries)
            fn.SetParLimits(1, 0., mean_est * 2)
        elif self.seed == 'file':
            with open(seed_file) as json_file:
                seed_info = json.load(json_file)
                for i in range(fn.GetNpar()):
                    fn.SetParameter(i, seed_info['pars']['%i' % (i)])
                    fn.SetParName(i, seed_info['names']['%i' % (i)])
    
    def jsonify(self,fit_info=""):
        if fit_info == "":
            eta_start = self.file_name.find('eta')
            num_match = re.search("\\d+(?=\\.root)",self.file_name[eta_start:])
            num = int(num_match.group())
            num_start = int(num_match.start())
            padded_num_name = re.sub("\\d+(?=\\.json)","%04i"%(num),self.file_name)

            print "1",os.getcwd()[:os.getcwd().rfind('/')]
            pname_dir = os.getcwd()[:os.getcwd().rfind('/')]+"/out/"+self.pname
            fit_name_dir = pname_dir+"/"+self.fit_name
            #pname directory
            if not os.path.isdir(pname_dir):
                os.mkdir(pname_dir)
                print "2",pname_dir
            #fit_name directory
            if not os.path.isdir(fit_name_dir):
                os.mkdir(fit_name_dir)
                print "3",fit_name_dir
            
            fit_info = fit_name_dir+"/%sfitter-%s-%s-%s.json" % (
                    "norm" if self.normalized else "",self.fit_name,self.pname,
                    self.file_name[eta_start:eta_start + num_start] + "%04i"%(num))
            
            print fit_info
        
        info = {}

        info['file_name'] = self.file_name
        info['fit_name'] = self.fit_name
        info['pname'] = self.pname
        info['cuts'] = self.cuts
        info['var'] = self.var
        info['bins'] = "%i,%f,%f"%(self.bins,self.lo,self.hi)
        par_names = list(self.func.GetParName(i)
                        for i in range(self.func.GetNpar()))
        par_values = list(self.func.GetParameter(i)
                        for i in range(self.func.GetNpar()))
        pars = {}
        names = {}

        for i in range(self.func.GetNpar()):
            pars.update({i: self.func.GetParameter(i)})
            names.update({i: self.func.GetParName(i)})

        info['normalized'] = self.normalized
        info['pars'] = pars
        info['names'] = names
        info['chi'] = self.chi
        info['NDF'] = self.NDF
        info['chiprob'] = self.chiprob
        
        with open(fit_info, 'w') as json_out:
            json.dump(info, json_out, indent=4)
