import ROOT
import time
import random

from ROOT import Math
from ROOT import TMath

from fitter_v0 import fitter

if __name__ == '__main__':
    canv = ROOT.TCanvas("canv","title",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    fit_name = 'crys_ball'

    ftr = fitter(fit_name)
    func = ftr.get_fit()
    hist = ftr.hist
    hist.Draw()
    func.Draw("same")
    # print func.Eval(90)
    # print func.Eval(275)
    # print func.Eval(475)
    # print func.Eval(725)
    # print func.Eval(950)

    lgn = ROOT.TLegend(0.675,0.75,0.875,0.875)
    lgn.AddEntry("hist","fit data","f")
    lgn.AddEntry("func","%s fit"%(ftr.fit_name),"l")
    
    if fit_name is 'dbl_gaus':
        g1 = ROOT.TF1('gaus1','gaus',ftr.lo,ftr.hi)
        g1.SetParameters(func.GetParameter(0),func.GetParameter(1),func.GetParameter(2))
        g1.SetLineColor(3)
        lgn.AddEntry('g1','gaus1','l')
        g2 = ROOT.TF1('gaus2','gaus',ftr.lo,ftr.hi)
        g2.SetParameters(func.GetParameter(3),func.GetParameter(4),func.GetParameter(5))
        g2.SetLineColor(7)
        lgn.AddEntry('g2','gaus2','l')
        
        g1.Draw("same")
        g2.Draw("same")

    lgn.Draw("same")

    par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
    par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
    pars = zip(par_names,par_values)
    #print pars
    #print ftr

    # for x in range(0,100,1):
    #     print "\tf(%0.3f) = %0.3f"%(x/100.,func.Eval(x/100.))
    #     print "intg(f(%0.3f)dx) = %0.3f"%(x/100.,func.Integral(ftr.lo,x/100.))

    # print "\tf(%0.3f) = %0.3f"%(5.,func.Eval(5.))
    # print "intg(f(%0.3f)dx) = %0.3f"%(5.,func.Integral(ftr.lo,5.))

    #canv.Clear()
    #hist.GetCumulative().Draw("same")

    # cum = ftr.get_cum()
    # cum_par_names = list(cum.GetParName(i) for i in range(cum.GetNpar()))
    # cum_par_values = list(cum.GetParameter(i) for i in range(cum.GetNpar()))
    # cum_pars = zip(cum_par_names,cum_par_values)
    # print cum
    # cum.SetLineColor(10)
    # cum.Draw("same")

    cmd = " "
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        time.sleep(3)
        cmd = raw_input()
        if cmd == 's':
            affirm = raw_input("are you sure you want to save this model? [y/n]")
            if affirm == 'y':
                ftr.jsonify()