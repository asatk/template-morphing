import ROOT
import json
import numpy as np
from matplotlib import pyplot as plt

from defs import PROJECT_DIR
from defs import file_path_check

plt.ion()

hs = ROOT.THStack("hs","2D Moment Morph")

# drawing variables and cuts
phi_var = "Obj_PhotonTwoProng.mass"
omega_var = "TwoProng_MassPi0[0]"


mass_list = [
    (300,0.2),(600,0.2),(900,0.2),(1200,0.2),(1500,0.2),(1800,0.2),(2100,0.2),(2400,0.2),(2700,0.2),(3000,0.2),
    (300,0.4),(600,0.4),(900,0.4),(1200,0.4),(1500,0.4),(1800,0.4),(2100,0.4),(2400,0.4),(2700,0.4),(3000,0.4),
    (300,0.6),(600,0.6),(900,0.6),(1200,0.6),(1500,0.6),(1800,0.6),(2100,0.6),(2400,0.6),(2700,0.6),(3000,0.6),
    (300,0.8),(600,0.8),(900,0.8),(1200,0.8),(1500,0.8),(1800,0.8),(2100,0.8),(2400,0.8),(2700,0.8),(3000,0.8),
    (300,1.0),(600,1.0),(900,1.0),(1200,1.0),(1500,1.0),(1800,1.0),(2100,1.0),(2400,1.0),(2700,1.0),(3000,1.0),
    (300,1.2),(600,1.2),(900,1.2),(1200,1.2),(1500,1.2),(1800,1.2),(2100,1.2),(2400,1.2),(2700,1.2),(3000,1.2),
    (300,1.4),(600,1.4),(900,1.4),(1200,1.4),(1500,1.4),(1800,1.4),(2100,1.4),(2400,1.4),(2700,1.4),(3000,1.4),
    (300,1.6),(600,1.6),(900,1.6),(1200,1.6),(1500,1.6),(1800,1.6),(2100,1.6),(2400,1.6),(2700,1.6),(3000,1.6),
    (300,1.8),(600,1.8),(900,1.8),(1200,1.8),(1500,1.8),(1800,1.8),(2100,1.8),(2400,1.8),(2700,1.8),(3000,1.8),
    (300,2.0),(600,2.0),(900,2.0),(1200,2.0),(1500,2.0),(1800,2.0),(2100,2.0),(2400,2.0),(2700,2.0),(3000,2.0),
]

sig_x = 30*1/4
sig_y = 0.02*1/4

xbins,xlo,xhi,ybins,ylo,yhi = (300,0,3000,200,0,2.000)

data_label_map = {}
data_mapping_path = "data_mapping.json"

# estimate 1st and 2nd gaussian moments for each distribution
for i in range(len(mass_list)):

    xy_gaus_i = ROOT.TF2("xy_gaus_i","xygaus",xlo,xhi,ylo,yhi)
    xy_gaus_i.SetNpx(xbins)
    xy_gaus_i.SetNpy(ybins)
    xy_gaus_i.SetParameters(1.,mass_list[i][0],sig_x,mass_list[i][1],sig_y)
    hist = xy_gaus_i.CreateHistogram()
    hist.Scale(1/hist.GetMaximum())
    
    # for j in range(xy_gaus_i.GetNpar()):
    #     print xy_gaus_i.GetParName(j),xy_gaus_i.GetParameter(j)
    
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


    mass_pair = [mass_list[i][0],mass_list[i][1]]
    mass_pair_str = "%04.0f,%1.2f"%(mass_pair[0],mass_pair[1])
    out_file_fstr = PROJECT_DIR+"/out/%s/gaus_%s.%s" 
    out_file_jpg = out_file_fstr%("hist_jpg",mass_pair_str,"jpg")
    out_file_png = out_file_fstr%("hist_png",mass_pair_str,"png")
    out_file_npy = out_file_fstr%("hist_npy",mass_pair_str,"npy")

    # hist.Draw("COL")
    # c1 = ROOT.gROOT.FindObject("c1")
    # c1.Update()
    # c1.SaveAs(out_file_jpg)    
    
    # plt.imsave(out_file_png,arr.T,cmap="gray",vmin=0.,vmax=1.,format="png",origin="lower")
    np.save(out_file_npy,arr,allow_pickle=False)

    data_label_map.update({out_file_npy:mass_pair})
    # temp=raw_input()

    # c1.Clear()

with open(data_mapping_path, 'w') as json_file:
    json.dump(data_label_map, json_file, indent=4)
