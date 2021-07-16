import ROOT
import time
import json
import os
import sys
import re

from collections import deque

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

from defs import PROJECT_DIR

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

class viewer:

    def __init__(self,fit_info_list,q=None):
        self.q = deque() if q is None else q
        self.fit_info_list = fit_info_list
        self.ftr = self.__fitter()
        #set up canvas
        self.canv = ROOT.TCanvas("canv","plot analysis",1200,900)
        self.canv.DrawCrosshair()
        self.canv.cd()


    def view(self):
        cmd = " "
        while cmd != "":
            if len(self.q) > 0:
                cmd = self.q.pop()
            else:
                cmd = raw_input("cmd: ")
            
            if cmd == 'p':
                print "[PARAMETERS LIST]"
                self.list_pars()
            elif cmd == 's':
                print "[SAVING]"
                if len(self.q) > 0:
                    opt = self.q.pop()
                else:
                    opt = raw_input("Type of view (SAME, SPLIT)")
                self.__save(opt=opt)
            elif cmd == 'v':
                print "[VIEWING]"
                if len(self.q) > 0:
                    opt = self.q.pop()
                else:
                    opt = raw_input("Type of view (SAME, SPLIT)")
                self.view_fits(opt=opt)

    def view_fits(self,opt="SPLIT"):
        #create list of mass points
        mass_pts = self.__mass_pts()    
        num_mass_pts = len(mass_pts)
        num_fits = len(self.fit_info_list)

        if opt == "SAME":
            self.canv.DrawFrame(0,0,5,0.5)
        elif opt == "SPLIT":
            self.canv.Divide(num_fits//num_mass_pts,num_mass_pts,0,0)

        lgn = self.__legend()

        #iterate through and draw each fit
        for count,i in enumerate(self.fit_info_list):
            # set up fitter and fit info
            # print "Viewer - using fit model #%i: %s"%(count,i)

            #create fitter for each fit
            ftr = fitter(i)

            hist = ftr.func.GetHistogram().Clone()
            color = ROOTCOLORS2[count%(num_fits//num_mass_pts)] - 3
            hist.SetLineColor(color)
            hist.SetFillColor(color)
            hist.SetFillStyle(3003)
            self.canv.cd(count+1 if opt == "SPLIT" else 0)
            hist.Draw("SAME")
            ROOT.gROOT.GetListOfSpecials().Add(hist)

            mass_idx = i.find('PH-')
            ext_idx = i[mass_idx:].find('.')

            lgn_func = lgn.AddEntry(hist,"%s - %4.4f"%(i[mass_idx:mass_idx + ext_idx],ftr.chi),"l")
            lgn_func.SetLineWidth(4)
            lgn_func.SetLineColor(hist.GetFillColor())

        lgn.Draw()
        ROOT.gROOT.GetListOfSpecials().Add(lgn)
        self.canv.Update()

    def list_pars(self):
        for count,i in enumerate(self.fit_info_list):
            # set up fitter and fit info
            print "using fit model #%i: %s"%(count,i)

            ftr = fitter(i)
            func = ftr.func.Clone()

            # print "p - [FETCHING PARAMETERS]"
            par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
            par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
            pars = zip(par_names,par_values)
            print str(par_values)[1:-1] + "\n"
            # print par_names
            # print "p - ",pars
            # print "p - Chi-Squared",ftr.chi

    def __mass_pts(self):
        mass_pts = set()
        for f in self.fit_info_list:
            if self.ftr.pname == 'phi':
                mass_pt = float(re.search("PH-(\d{4}).*(?=\.json)", f).group(1))
            elif self.ftr.pname == 'omega':
                om_text = re.search("OM-(\d)p(\d{3}).*(?=\.json)", f)
                mass_pt = float(om_text.group(1)) + float(om_text.group(2))/1000
            mass_pts.add(mass_pt)
        return mass_pts

    def __fitter(self):
        #create fitter
        print self.fit_info_list[0]
        return fitter(self.fit_info_list[0])

    def __legend(self):
        lgn = ROOT.TLegend(0.55,0.55,0.89,0.88)
        lgn.SetTextFont(22)
        lgn.SetHeader("Fit Function Chi^2")
        lgn.SetTextFont(132)
        lgn.SetEntrySeparation(0.05)
        lgn.SetTextSize(0.025)
        lgn.SetLineWidth(2)
        lgn.SetFillColor(19)
        return lgn

    def __save(self,opt):
        run_name = re.search(r"(\w+)_PH-\d{4}",self.fit_info_list[0]).group(1)
        self.canv.SaveAs('../out/' + run_name + '-' +  opt.lower() +  '.jpg')

if __name__ == "__main__":
    print sys.argv
    print sys.path
    fit_info_list = json.load(open(os.getcwd()[:os.getcwd().rfind('src')+3]+'config.json','r'))['fit_info_list']
    vwr = viewer(fit_info_list)
    vwr.view_fits()
    vwr.list_pars()
    # view_fits()
    # list_fits()