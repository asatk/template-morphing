import sys
import ROOT
import math
import string
import time
import random

from ROOT import kRed
from ROOT import kOrange
from ROOT import kGreen
from ROOT import kBlue
from ROOT import Math
from ROOT import TMath




fileName = info['fname']

s = info['bins'].split(',')
bins = int(s[0])
lo = float(s[1])
hi = float(s[2])

user_data = info['user_data'] == 'y'
scale = info['scale'] == 'y'
cum = info['cum'] == 'y'
nfill = int(info['nfill'])
gaus_seed = info['seed'] == 'gaus'
offset = 1 if gaus_seed else 0

if user_data:
    chain = ROOT.TChain("twoprongNtuplizer/fTree")
    chain.Add(fileName)

mean_est = float(fileName[fileName.find("eta")+3:fileName.find(".root")])\
    if user_data else 0.

start = 0
end = 0

canv = ROOT.TCanvas("canv","title",1200,900)
#canv.DivideSquare(end-start+1+offset)
canv.DrawCrosshair()
canv.cd()

info = {}
fits = {}

def setup():
    global info,fits
    
    info = {
        'fname': 'root/TwoProngNtuplizer_eta500.root',
        'var': 'Obj_PhotonTwoProng.mass',
        #'var': 'TwoProng_MassPi0[0]',
        #'var': 'GenPhi_mass.GenPhi_mass',
        'bins': '100,0,2500',
        #'bins': '50,0,5',
        #'bins': '100,999.95,1000.05',
        #'bins': '50,-20,20',
        #'cuts': 'nTwoProngs>0 && nIDPhotons>0 && Obj_PhotonTwoProng.dR>0.1 && Obj_PhotonTwoProng.part2_pt>150'
        'cuts': 'nTwoProngs>0 && nIDPhotons>0 && Obj_PhotonTwoProng.dR>0.1 && Obj_PhotonTwoProng.part2_pt>150',
        'user_data':'y',
        'scale':'y',
        'cum':'n',
        'nfill':'10000',
        'seed':'',
        'fit_name':'gaus'
    }
    fits = {
        'gaus':'gaus',
        'gaus_pdf':'[0]*ROOT::Math::normal_pdf(x[0],[1],[2])',
        'gaus_cdf':'[0]*ROOT::Math::normal_cdf(x[0],[1],[2])',
        'dbl_gaus':'gaus(0)+gaus(3)',
        'dbl_gaus_cdf':'',
        'crys_ball':'[0]*ROOT::Math::crystalball_function(x[0],[1],[2],[3],[4])',
        'crys_ball_pdf':'[0]*ROOT::Math::crystalball_pdf(x[0],[1],[2],[3],[4])',
        'crys_ball_cdf':'[0]*ROOT::Math::crystalball_cdf(x[0],[1],[2],[3],[4])',
        'landau':'landau',
        'landau_cdf':'',
        'landxgaus':'',
        'landxgaus_cdf':''
    }
    return

def test_fit(fit_name="gaus"):
    global info,fits
    print "TESTING %s FIT"%(fit_name)
    hist = ROOT.TH1D("hist","plot %s fit"%(fit_name),bins,lo,hi)
    hist.SetFillColor(45)

    if fit_name in fits.keys():
        fit_str = fits[fit_name]
        print "USING FIT %s"%(fit_str)
    else:
        print "FIT NOT COMPATIBLE...exiting"
        exit

    fn = ROOT.TF1("fit",fit_str,lo,hi)

    if fit_name in ['gaus','gaus_pdf','gaus_cdf']:
        fn.SetParName(0,'Constant')
        fn.SetParName(1,'Mean')
        fn.SetParName(2,'Sigma')

    if user_data:
        draw_s = info['var'] + ">>hist"
        cut_s = info['cuts']
        chain.Draw(draw_s,cut_s)
    else:
        hist.FillRandom(fit_str,nfill)
        hist.Draw()
    if scale:
        hist.Scale(1/hist.Integral())
    if 'cum' in fit_str:
        hist = hist.GetCumulative()
        hist.Draw()
    
    
    fn.SetParameters(random.random()+0.5,random.randint(0,100),mean_est)

    fit_res = hist.Fit("fit","SM0")
    fit_fn = hist.GetFunction("fit")
    
    fit_fn.SetLineColor(4)
    fit_fn.Draw("same")

    #print fit_gaus_res
    #print fit_gaus.GetChisquare()
    #print gaus_fn.Eval(mean_est)
    #for i in range(gaus_fn.GetNpar()):
    #    print gaus_fn.GetParameter(i)
    #print fit_gaus.GetFCN()

    print("TEST %s - END"%(fit_name))

#Pad 2 - gaus Fill + Fit
if (start <= 2 and end >= 2) or gaus_seed:

    print "TEST #2 - GAUS"

    canv.cd(2-start+1+offset)
    hist2 = ROOT.TH1D("hist2","plot gaus",bins,lo,hi)

    hist2.SetFillColor(45)

    if user_data:
        draw_s = info['var'] + ">>hist2"
        cut_s = info['cuts']
        chain.Draw(draw_s,cut_s)
    else:
        hist2.FillRandom("gaus",nfill)
        hist2.Draw()
    if scale:
        hist2.Scale(1/hist2.Integral())
    if cum and not gaus_seed:
        gaus_cum = ROOT.TF1("gaus_cum","[0]*ROOT::Math::normal_cdf(x[0],[1],[2])",lo,hi)
        gaus_cum.SetParameters(random.random()+0.5,random.randint(0,100),mean_est)
        hist2_cum = hist2.GetCumulative()
        hist2_cum.Draw()
        
        fit_gaus_res = hist2_cum.Fit("gaus_cum","S0")
        fit_gaus = hist2_cum.GetFunction("gaus_cum")
    else:
        gaus_fn = ROOT.TF1("gaus_fn","[0]*ROOT::Math::normal_pdf(x[0],[1],[2])",lo,hi)
        gaus_fn.SetParameters(random.random()+0.5,random.randint(0,100),mean_est)

        fit_gaus_res = hist2.Fit("gaus_fn","S0")
        fit_gaus = hist2.GetFunction("gaus_fn")
    
    fit_gaus.SetLineColor(4)
    fit_gaus.Draw("same")

    #print fit_gaus_res
    #print fit_gaus.GetChisquare()
    #print gaus_fn.Eval(mean_est)
    #for i in range(gaus_fn.GetNpar()):
    #    print gaus_fn.GetParameter(i)
    #print fit_gaus.GetFCN()

    print "TEST #2 - END"

#Pad 3 - dbl_gaus Fill + Fit
if start <= 3 and end >= 3:

    print "TEST #3 - DBL_GAUS"

    canv.cd(3-start+1+offset)
    hist3 = ROOT.TH1D("hist3","plot dbl_gaus",bins,lo,hi)

    dbl_gaus = ROOT.TF1("dbl_gaus","gaus(0) + gaus(3)",lo,hi)
    if gaus_seed:
        gaus_pars = fit_gaus.GetParameters()
        dbl_gaus.SetParameters(\
            gaus_pars[0]*random.random(),gaus_pars[1],gaus_pars[2],\
            gaus_pars[0]*random.random(),gaus_pars[1],gaus_pars[2])
    else:
        dbl_gaus.SetParameters(10,5,1000,20,10,1000)
    dbl_gaus.SetLineColor(7)
    #dbl_gaus.Draw()

    if user_data:
        draw_s = info['var'] + ">>hist3"
        cut_s = info['cuts']
        chain.Draw(draw_s,cut_s)
    else:
        hist3.FillRandom("dbl_gaus",nfill)
        hist3.Draw()
    if scale:
        hist3.Scale(1/hist3.Integral())

    hist3.SetFillColor(44)

    hist3.Fit("dbl_gaus","SM0L")
    fit_dbl_gaus = hist3.GetFunction("dbl_gaus")
    fit_dbl_gaus.Draw("same")

    print "TEST #3 - END"

#Pad 4 - crys_ball Fill + Fit
if start <= 4 and end >= 4:

    print "TEST #4 - CRYS BALL"

    canv.cd(4-start+1+offset)
    hist4 = ROOT.TH1D("hist4","plot crys_ball",bins,lo,hi)

    crys = ROOT.TF1("crys","ROOT::Math::crystalball_function(x[0],[0],[1],[2],[3])",lo,hi)
    #crys_hist.SetParameters(0.25,1,15,500)
    #crys.SetParameters(2,1,1,0)
    #crys.SetLineColor(kGreen)
    #crys.Draw()
    #crys_hist.Draw()

    if gaus_seed:
        gaus_pars = fit_gaus.GetParameters()
        crys.SetParameters(random.random(),1,gaus_pars[2]*random.random(),gaus_pars[1])
    else:
        crys.SetParameters(0.25,1,2.5,500)
    crys.SetLineColor(kGreen)

    if user_data:
        draw_s = info['var'] + ">>hist4"
        cut_s = info['cuts']
        chain.Draw(draw_s,cut_s)
    else:
        hist4.FillRandom("crys_hist",nfill)
        hist4.Draw()
    if scale:
        hist4.Scale(1/hist4.Integral())

    hist4.SetFillColor(35)

    fit_crys_res = hist4.Fit("crys","SM0")
    fit_crys = hist4.GetFunction("crys")
    fit_crys.SetLineColor(kBlue)
    fit_crys.Draw("same")

    print fit_crys_res
    print fit_crys.GetChisquare()

    fit_log_crys_res = hist4.Fit("crys","SM0L+")
    fit_log_crys = hist4.GetFunction("crys")
    fit_log_crys.SetLineColor(kRed)
    fit_log_crys.Draw("same")

    print fit_log_crys_res
    print fit_log_crys.GetChisquare()

    print "TEST #4 - END"

#Pad 5 - crys_ball user Fill + Fit
if start <= 5 and end >= 5:

    print "TEST #5 - CRYS BALL USER"

    canv.cd(5-start+1+offset)
    hist5 = ROOT.TH1D("hist5","plot crys_ball_user",bins,lo,hi)

    def cball_fn(x,par):
        return ROOT.Math.crystalball_function(x[0],par[0],par[1],par[2],par[3])

    cb = ROOT.TF1("cb",cball_fn,lo,hi,4)
    cb.SetParameters(0.25,1,2.5,500)
    #cb.SetLineColor(6)
    print cb.Eval(5)
    #cb.Draw()

    if user_data:
        draw_s = info['var'] + ">>hist5"
        cut_s = info['cuts']
        chain.Draw(draw_s,cut_s)
    else:
        hist5.FillRandom("cb",nfill)
        hist5.Draw()
    if scale:
        hist5.Scale(1/hist5.Integral())

    hist5.SetFillColor(20)

    fit5 = hist5.Fit("cb","SU")
    fit_crys_user = hist5.GetFunction("cb")
    print fit5
    fit_crys_user.Draw()
    #hist5.Fit("ROOT::Math::crystalball_function(x)")

    x=5
    val = ROOT.Math.crystalball_function(x,0.25,1,2.5,15)
    print x,val

    print "TEST #5 - END"

#Pad 6 - user data Fit
if start <= 6 and end >= 6:
    
    print "TEST #6 - user data"

    canv.cd(5-start+1+offset)
    hist6 = ROOT.TH1D("hist6","plot data",bins,lo,hi)
    
    draw_s = info['var'] + ">>hist6"
    cut_s = info['cuts']

    chain.Draw(draw_s,cut_s)

    hist6.Scale(1/hist6.Integral())
    print hist6.Integral()
    hist6.SetFillColor(37)

    canv.Update()

    print hist6.Fit("gaus","S")
    fit_data_gaus = hist6.GetFunction("gaus")
    fit_data_gaus.SetLineColor(3)
    fit_data_gaus.SetLineWidth(5)

    gaus_const = fit_data_gaus.GetParameter(0)
    gaus_mean = fit_data_gaus.GetParameter(1)
    gaus_sigma = fit_data_gaus.GetParameter(2)

    print gaus_const,gaus_mean,gaus_sigma

    dbl_gaus = ROOT.TF1("dbl_gaus","gaus(0) + gaus(3)",lo,hi)
    dbl_gaus.SetParameters(gaus_const*0.5,gaus_mean*0.5,gaus_sigma*0.5,gaus_const*0.5,gaus_mean*0.5,gaus_sigma*0.5)
    dbl_gaus.SetLineColor(4)
    #crys = ROOT.TF1("crys","ROOT::Math::crystalball_function(x,[0],[1],[2],[3])",lo,hi)
    #crys.SetParameters(0.5,1,0,1)
    #land = ROOT.TF1("land","TMath::Landau(x,[0],[1])",lo,hi)

    def land_fn(x,par):
        return par[2]*TMath.Landau(x[0],par[0],par[1])
    land = ROOT.TF1("land",land_fn,lo,hi,3)
    land.SetParameters(1,0.3,1)

    print hist6.Fit("dbl_gaus","S+")
    fit_data_dbl_gaus = hist6.GetFunction("dbl_gaus")

    #print hist6.Fit("crys","S+")
    #fit_data_crys = hist6.GetFunction("crys")

    #print hist6.Fit("land","SU+")
    #fit_data_land_user = hist6.GetFunction("land")

    #print hist6.Fit("landau","S+")
    #fit_data_landau = hist6.GetFunction("landau")

if __name__ == '__main__':
    setup()
    test_fit(info['fit_name'])
    cmd = " "
    while not cmd == "" and\
        not cmd[(cmd.rfind('/') if cmd.rfind('/') != -1 else 0):] == "/fit_v1.py":
        time.sleep(3)
        cmd = raw_input()