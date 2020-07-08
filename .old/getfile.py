import sys
import time
import ROOT

fileName = "root/TwoProngNtuplizer_eta125.root"
file = ROOT.TFile(fileName)

canv = ROOT.TCanvas()
canv.cd()

info = {
    'var': 'Obj_PhotonTwoProng.mass',
    #'var': 'TwoProng_MassPi0[0]',
    'bins': '50,0,2000',
    #'bins': '50,0,5',
    'cuts': 'nTwoProngs>0 && nIDPhotons>0 && Obj_PhotonTwoProng.dR>0.1 && Obj_PhotonTwoProng.part2_pt>150'
}

s = info['bins'].split(',')
bins = int(s[0])
lo = float(s[1])
hi = float(s[2])

hist = ROOT.TH1D("hist_a","hist title",bins,lo,hi)

chain = ROOT.TChain("twoprongNtuplizer/fTree")
chain.Add(fileName)

draw_s = info['var'] + ">>hist_a"
cut_s = info['cuts']

chain.Draw(draw_s,cut_s)

cmd = " "
while not cmd == "" and\
    not cmd == "/home/asatk/miniconda3/envs/cern2.7/bin/python /home/asatk/Documents/code/cern/TM/getfile.py":
    time.sleep(3)
    cmd = raw_input("")