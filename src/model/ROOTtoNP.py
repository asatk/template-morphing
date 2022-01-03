#!/home/asatk/miniconda3/envs/cern2.7/bin/python

import ROOT
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
# os.system("which python")

from defs import PROJECT_DIR
# print("project dir: "+PROJECT_DIR)

file_list = sys.argv[1:]

if not os.path.isdir(PROJECT_DIR+"/out/"):
    os.mkdir(PROJECT_DIR+"/out/")
    os.mkdir(PROJECT_DIR+"/out/jpg")
    os.mkdir(PROJECT_DIR+"/out/png")
    os.mkdir(PROJECT_DIR+"/out/npy")

plt.ion()

hs = ROOT.THStack("hs","2D Moment Morph")

# drawing variables and cuts
phi_var = "Obj_PhotonTwoProng.mass"
omega_var = "TwoProng_MassPi0[0]"
draw_str = omega_var+":"+phi_var+">>hist"
cut_str = "nTwoProngs>0 && nIDPhotons>0 && Obj_PhotonTwoProng.dR>0.1 && Obj_PhotonTwoProng.part2_pt>150"

# Tree name in Ntuplizer files
tree_name = "twoprongNtuplizer/fTree"

xbins,xlo,xhi,ybins,ylo,yhi = (300,0,3000,200,0,2.000)

# estimate 1st and 2nd gaussian moments for each distribution
for i,f in enumerate(file_list):

    hist = ROOT.TH2D("hist","hist",xbins,xlo,xhi,ybins,ylo,yhi)
    
    # draw .root data into histogram
    chain = ROOT.TChain(tree_name)
    chain.Add(f)
    chain.Draw(draw_str,cut_str,"goff")
    # hist = hist.DrawNormalized("0")

    # # scale histogram from 0.0 to 1.0
    # hist.Scale(1/hist.GetMaximum())

    # normalize histogram to the num of entries in cut
    hist.Scale(1/hist.Integral())
    
    # make np arrays to store, use in py3 with keras
    x = np.zeros(xbins)
    y = np.zeros(ybins)
    arr = np.zeros((xbins,ybins))

    for bin_y in range(ybins):
        y_val = hist.GetYaxis().GetBinLowEdge(bin_y)
        y[bin_y] = y_val
        for bin_x in range(xbins):
            x_val = hist.GetXaxis().GetBinLowEdge(bin_x)
            x[bin_x] = x_val
            z_val = hist.GetBinContent(bin_x,bin_y)
            # don't access np array if unecessary
            if z_val != 0:
                arr[bin_x,bin_y] = z_val

    file_str = file_list[i]
    idx_slash = file_str.rfind('/')
    idx_radix = file_str.rfind('.')
    file_str = file_str[idx_slash+1:idx_radix]
    print("file str: "+file_str)

    out_file_fstr = PROJECT_DIR+"/out/{suffix}/"+file_str+".{suffix}"
    out_file_jpg = out_file_fstr.format(suffix="jpg")
    out_file_png = out_file_fstr.format(suffix="png")
    out_file_npy = out_file_fstr.format(suffix="npy")

    hist.Draw("COL")
    c1 = ROOT.gROOT.FindObject("c1")
    c1.Update()
    c1.SaveAs(out_file_jpg)

    fig,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(10,5),dpi=100)
    ax1.set(xlim=(xlo,xhi),ylim=(ylo,yhi))
    ax2.set(xlim=(xlo,xhi),ylim=(ylo,yhi))

    ax1.contour(x,y,arr.T,50)
    ax1.set_title("contour")
    
    im=ax2.imshow(arr.T,cmap="gray",vmin=0.,vmax=1.,
            extent=[xlo,xhi,ylo,yhi],
            origin="lower",aspect="auto")
    ax2.set_title("grayscale contour 0. to 1.")
    
    plt.show()
    
    plt.imsave(out_file_png,arr.T,cmap="gray",vmin=0.,vmax=1.,format="png",origin="lower")
    np.save(out_file_npy,arr,allow_pickle=False)

    # temp=raw_input()

    plt.close("all")
    c1.Clear()
    hist.Delete()