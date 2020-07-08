import ROOT
import sys
import string
import math
from scipy import special as spec

from ROOT import kRed
from ROOT import kGreen
from ROOT import kBlue
from ROOT import TF1
from ROOT import gROOT
from ROOT import Math
from ROOT.Math import crystalball_function

# inputName = sys.argv[1]
# varName = sys.argv[3]
# binstr = sys.argv[5].split(',')
# bins = int(binstr[0])
# lo = float(binstr[1])
# hi = float(binstr[2])
# cuts = sys.argv[7]

lo = -20
hi = 20
bins = 50


hist = ROOT.TH1D("hist"+"_sum_"+str(1), "hist", bins, lo, hi)
hist2 = ROOT.TH1D("hist"+"_sum_"+str(2), "hist", bins, lo, hi)
#hist2.FillRandom("gaus",10000)
#hist2.Draw()

# chain = ROOT.TChain("twoprongNtuplizer/fTree")
# chain.Add(inputName)
# tree = chain.CopyTree(cuts)
# nevents = chain.GetEntries()
# event = chain.SetBranchAddress("event")
# nevents = 
# print nevents
# for i in range(nevents):
#     event = chain.GetEvent(i)
#     hist.Fill(event)
# chain.ls()

def crystal_ball(x,par):
    alpha = par[0]
    alpha_abs = abs(par[0])
    n = ROOT.Int_t(par[1])
    mean = par[2]
    sigma = par[3]


    z = (x[0]-mean)/sigma

    if sigma < 0.: return 0.
    #if alpha < 0.: z = -z

    if z > -1*alpha :
        return math.exp(-0.5*pow(z,2))

    #print "alpha: %s\t alpha_abs %s\t n: %s\t mean: %s\t sigma: %s\t z: %s\n"\
    #    %(alpha,alpha_abs,n,mean,sigma,z)

    A = pow((n/alpha_abs),n)*math.exp(-1*pow(alpha_abs,2)/2)
    B = n/alpha_abs - alpha_abs
    C = (n/alpha_abs)*(1/(n-1))*math.exp(-1*pow(alpha_abs,2)/2)
    D = math.sqrt(math.pi/2)*(1+spec.erf(alpha_abs/math.sqrt(2)))
    N = 1/(sigma*(C+D))
    
    #print "A: %s\t B: %s\t C: %s\t D: %s\t N: %s\n"%(A,B,C,D,N)

    return A*pow((B-z),-1*n)

print crystal_ball([1.0],[-10.0,2.0,0.0,1.0])

cball = TF1('crystal_ball',crystal_ball,-20,20,4)
#cball.SetParNames("alpha","n","mean","sigma")
cball.SetParameters(1.0,2.0,0.0,1.0)
#cball.SetLineColor(kRed)
#cball.Draw()
#hist2.Fit("gaus")
#hist2.Fit("crystal_ball")
hist_cball = ROOT.TH1F("cball fill","Fill CBall FN Randomly",200,-20,20)
hist_cball.SetFillColor(45)
hist_cball.FillRandom("crystal_ball",1000000)
hist_cball.Draw()
#canv = ROOT.TCanvas("canv","plotting",200,10,700,500)
#canv.cd(1)

#for i in range(lo,hi,1):
    #print eval(str(i)+'.')
    #print cball.Eval(eval(str(i)+'.'))



#chain.Draw("Obj_PhotonTwoProng.mass")

#f1 = ROOT.TF1("f1","gaus",-5,5)
#f1.SetParameters(0,1)
#f1.SetLineColor(kRed)
#f1.Draw()

cmd = " "
while not cmd == "":
    session_input = raw_input("[Enter] to quit")
    cmd = session_input

canv = ROOT.TCanvas("canv","title")
canv.DivideSquare(4)
canv.cd(1)
canv.DrawCrosshair()

#crys = TF1("crys","ROOT.Math.crystalball_function(x,2,2,1,0)",-5,5)
#crys = TF1("crys","crystalball_function(x,2,2,1,0)",-5,5)
#crys = TF1("crys",ROOT.Math.crystalball_function,-5,5)
#crys.SetLineColor(kGreen)
#hist_crys = ROOT.TH1D("ROOT crys","Fill ROOT.Math.crystalball_function",bins,lo,hi)
#hist_crys = ROOT.TH1D("ROOT crys","Fill crystalball_function",bins,lo,hi)
#crys.Draw()
#hist_crys.FillRandom("ROOT.Math.crystalball_function",10000)
hist_crys = ROOT.TH1D("ROOT crys","user crystalball fit",bins*5,lo,hi)
hist_crys.FillRandom("crystal_ball",100000)
hist_crys.SetFillColor(35)
hist_crys.Draw()
#hist_crys.Fit(cball,"U")
#hist_crys.Fit("crystal_ball","U")
#hist.Fit("ROOT.Math.crystalball_function")
#hist.Fit(ROOT.Math.crystalball_function,"U")



cmd = " "
while not cmd == "":
    session_input = raw_input("[Enter] to quit")
    cmd = session_input