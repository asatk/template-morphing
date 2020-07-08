import ROOT
import time
import random

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

def make_fit():
    canv = ROOT.TCanvas("canv","title",1200,900)
    canv.DrawCrosshair()
    canv.cd()

    fit_name = 'landau'
    fit_info = 'fitter-init.json'

    ftr = fitter(fit_name,fit_info)
    func = ftr.get_fit()
    hist = ftr.hist
    func.SetLineColor(kRed)
    func.SetLineWidth(3)
    hist.SetFillColor(kAzure+4)
    hist.Draw("HIST same")
    func.Draw("same")
    
    #func.DrawIntegral("same")

    print "intg func",func.Integral(ftr.lo,ftr.hi)
    #print "intg func",func.DrawIntegral("same").Eval(4.9)

    lgn = ROOT.TLegend(0.675,0.8,0.875,0.875)
    lgn.AddEntry("hist","fit data","F")
    lgn.AddEntry("func","%s fit"%(ftr.fit_name),"L")

    lgn.Draw("same")

    cmd = " "
    delay = 3.
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/run_fitter.py":
        cmd = raw_input("cmd:")
        if cmd == 's':
            print "[SAVING]"
            affirm = raw_input("are you sure you want to SAVE this model? [y/n]")
            if affirm == 'y':
                ftr.jsonify()
        elif cmd == 'r' or cmd == 'refit':
            print "[REFITTING]"
            affirm = raw_input("are you sure you want to REFIT? [y/n]")
            if affirm == 'y':
                func = ftr.get_fit()
                hist = ftr.hist
                func.SetLineColor(kRed)
                func.SetLineWidth(3)
                hist.SetFillColor(kAzure+4)
                hist.Draw()
                func.Draw("same")
        elif cmd == 'rebin':
            print "[REBINNING]"
            tempbins = raw_input("new bin size: ")
            templo = raw_input("new lo: ")
            temphi = raw_input("new hi: ")
            if tempbins is not '':
                ftr.bins = int(tempbins)
            if templo is not '':
                ftr.lo = float(templo)
            if temphi is not '':
                ftr.hi = float(temphi)
        elif cmd == 't':
            print "[DELAY]"
            delay = float(raw_input("New Delay Time (in sec):"))
        time.sleep(delay)

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
        cmd = raw_input("cmd:")
        time.sleep(3)

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
        cmd = raw_input("cmd:")
        if cmd == 's':
            print "[SAVING]"
            affirm = raw_input("are you sure you want to SAVE this model? [y/n]")
            if affirm == 'y':
                ftr.jsonify()
        elif cmd == 'd':
            print "[DRAWING]"
            func.Draw()
        elif cmd == 'i':
            print "[INTEGRATING]"
            intglo = float(raw_input("integrate from: "))
            intghi = float(raw_input("integrate to: "))
            print ftr.func.Integral(intglo,intghi)
            print func.Integral(ftr.lo,ftr.hi)
            #ftr.func.DrawIntegral()
            #print ftr.func.CreateHistogram().Integral()
        elif cmd == 'n':
            print "[NORMALIZING]"
            newfunc = func.Clone()
            newfunc.SetParameter(0,func.GetParameter(0)/func.Integral(ftr.lo,ftr.hi))
            newfunc.Draw()
            print newfunc.Integral(ftr.lo,ftr.hi)
        time.sleep(3)


if __name__ == '__main__':
    make_fit()
    #test()
    #get_fit()