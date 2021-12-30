#!/home/asatk/miniconda3/envs/cern2.7/bin/python

import ROOT
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

os.system("echo $PYTHONPATH")
os.system("pwd")
sys.path.append(os.getcwd())
os.system("echo $PYTHONPATH")
os.system("which python")

from defs import PROJECT_DIR
print "project dir:",PROJECT_DIR
# from defs import file_path_check

file_list = []

def __main__():
    global file_list
    print(sys.argv[0])
    print(sys.argv[1])
    file_list = sys.argv[1:]
    print "main"

if __name__ == "__main__":
    __main__()

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

# file_list = [
#     "${PROJECT_DIR}/root/Feb2021_PH-0500_OM-0p550.root",
#     "${PROJECT_DIR}/root/Feb2021_PH-0500_OM-0p950.root",
#     # "${PROJECT_DIR}/root/Feb2021_PH-1000_OM-0p550.root",
#     # "${PROJECT_DIR}/root/Feb2021_PH-1000_OM-0p950.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-0p290.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-0p420.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-0p480.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-0p650.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-0p750.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-0p850.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-1p100.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-1p300.root",
#     "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-1p500.root"
# ]

# mass_list = [
#     (500,0.55),
#     (500,0.95),
#     # (1000,0.55),
#     # (1000,0.95),
#     (500,0.29),
#     (500,0.42),
#     (500,0.48),
#     (500,0.65),
#     (500,0.75),
#     (500,0.85),
#     (500,1.1),
#     (500,1.3),
#     (500,1.5)
# ]

xbins,xlo,xhi,ybins,ylo,yhi = (300,0,3000,200,0,2.000)

# if not (len(file_list) == len(mass_list)):
#     print("list of root files and list of mass points are not the same length:")
#     print("file_list: %i\tmass_list: %i"%(len(file_list),len(mass_list)))
#     quit()
# validate that all root files exist
# for i in range(len(file_list)):
#     file_list[i] = file_path_check(file_list[i])


# estimate 1st and 2nd gaussian moments for each distribution
for i,f in enumerate(file_list):

    hist = ROOT.TH2D("hist","hist",xbins,xlo,xhi,ybins,ylo,yhi)
    
    # draw .root data into histogram
    chain = ROOT.TChain(tree_name)
    chain.Add(f)
    chain.Draw(draw_str,cut_str,"goff")

    # scale histogram from 0.0 to 1.0
    hist.Scale(1/hist.GetMaximum())
    
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

    # out_file_fstr = PROJECT_DIR+"/out/%s/%4.0f,%1.2f.%s" 
    # out_file_jpg = out_file_fstr%("hist_jpg",mass_list[i][0],mass_list[i][1],"jpg")
    # out_file_png = out_file_fstr%("hist_png",mass_list[i][0],mass_list[i][1],"png")
    # out_file_npy = out_file_fstr%("hist_npy",mass_list[i][0],mass_list[i][1],"npy")

    file_str = file_list[i]
    idx_slash = file_str.rfind('/')
    idx_radix = file_str.rfind('.')
    file_str = file_str[idx_slash+1:idx_radix]
    print "file str:",file_str

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




