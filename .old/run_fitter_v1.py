import ROOT
import time
import random

from ROOT import Math
from ROOT import TMath

from fitter import fitter

def make_fit():
    canv = ROOT.TCanvas("canv","title",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    fit_name = 'landau_pdf'
    fit_info = 'fitter-init.json'
    #fit_info = 'fitter-phi500.json'

    ftr = fitter(fit_name,fit_info)
    func = ftr.get_fit()
    hist = ftr.hist
    #hist.Draw("HIST")
    func.Draw("same")
    func.DrawIntegral("same")

    print "intg",func.Integral(ftr.lo,ftr.hi)
    print "intg func",func.DrawIntegral().Eval(5.)

    lgn = ROOT.TLegend(0.675,0.75,0.875,0.875)
    lgn.AddEntry("hist","fit data","f")
    lgn.AddEntry("func","%s fit"%(ftr.fit_name),"l")

    lgn.Draw("same")

    par_names = list(func.GetParName(i) for i in range(func.GetNpar()))
    par_values = list(func.GetParameter(i) for i in range(func.GetNpar()))
    pars = zip(par_names,par_values)

    cmd = " "
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        time.sleep(3)
        cmd = raw_input("cmd:")
        if cmd == 's':
            affirm = raw_input("are you sure you want to save this model? [y/n]")
            if affirm == 'y':
                ftr.jsonify()

def test():
    f1 = ROOT.TF1("f1","ROOT::Math::landau_pdf(x[0],[1],[0])",0.,5.)
    f1.SetParameters(0.68,0.074)
    #f1.Draw()

    f2 = ROOT.TF1("f2",'landau',0.,5.)
    f2.SetParameters(1,0.68,0.074)
    f2.SetLineColor(4)
    #f2.Draw("same")

    hist = ROOT.TH1D("hist","plotplot",100,-5.,5.)
    hist.FillRandom('f2',100000)
    hist.Draw("same")
    #time.sleep(3)
    # hist.Scale(1./hist.Integral())
    # hist.SetLineColor(8)
    # hist.Draw("same")
    # time.sleep(3)

    hist.Fit('landau','SM')
    func = hist.GetFunction('landau')
    print func.Integral(-5.,5.)
    func.SetParameter('Constant',func.GetParameter('Constant')/func.Integral(-5.,5.))
    func.SetLineColor(20)
    func.Draw()
    print func.Integral(-5.,5.)
    func.SetName('fit')
    func.SetTitle('fit')
    print func

    hist2 = ROOT.TH1D("hist2","plotplot-more",100,-5.,5.)
    hist2.FillRandom('fit',100000)
    hist2.Draw("same")
    
    cmd = " "
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        time.sleep(3)
        cmd = raw_input("cmd:")

def get_fit():
    canv = ROOT.TCanvas("canv","title",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    #fit_name = 'crys_ball'
    #fit_info = 'fitter-init.json'
    fit_info = 'fitter-phi-etaprime500.json'

    ftr = fitter(fitted=True,fit_info=fit_info)
    func = ftr.func
    #hist = ftr.hist
    #hist.Draw()
    func.Draw()

    lgn = ROOT.TLegend(0.675,0.75,0.875,0.875)
    #lgn.AddEntry("hist","fit data","f")
    lgn.AddEntry("func","%s fit"%(ftr.fit_name),"l")

    lgn.Draw("same")

    cmd = " "
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        time.sleep(3)
        cmd = raw_input("cmd:")
        if cmd == 's':
            affirm = raw_input("are you sure you want to save this model? [y/n]")
            if affirm == 'y':
                ftr.jsonify()
        if cmd == 'i':
            #intglo = float(raw_input("integrate from: "))
            #intghi = float(raw_input("integrate to: "))
            #print ftr.func.Integral(intglo,intghi)
            ftr.func.DrawIntegral()
            print ftr.func.CreateHistogram().Integral()


if __name__ == '__main__':
    make_fit()
    #test()
    #get_fit()