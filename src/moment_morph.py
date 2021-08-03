import json

import ROOT

from defs import PROJECT_DIR
from defs import file_path_check

def estimate_moments():

    c = ROOT.TCanvas("c","2D plot",600,600)
    c.cd()

    output_file = PROJECT_DIR+"/out/moments/2DGaus.json"

    # histogram parameters
    
    xbins,xlo,xhi,ybins,ylo,yhi = (300,0,3000,200,0,2.000)

    # xbins = 100
    # xlo = 0
    # xhi = 1000
    # ybins = 100
    # ylo = 0.25
    # yhi = 1.5
    draw_opt = "LEGO10"

    hist_2D_tot = ROOT.TH2D("hist2D_tot","2D Feb2021 Phi=500 Omega=0.55",xbins,xlo,xhi,ybins,ylo,yhi)
    hs = ROOT.THStack("hs","2D Moment Morph")


    # drawing variables and cuts
    phi_var = "Obj_PhotonTwoProng.mass"
    omega_var = "TwoProng_MassPi0[0]"
    draw_str = omega_var+":"+phi_var+">>hist_2D_i"
    cut_str = "nTwoProngs>0 && nIDPhotons>0 && Obj_PhotonTwoProng.dR>0.1 && Obj_PhotonTwoProng.part2_pt>150"

    # Tree name in Ntuplizer files
    tree_name = "twoprongNtuplizer/fTree"

    file_list = [
        "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-0p480.root",
        "${PROJECT_DIR}/root/Mar2021_PH-0500_OM-1p100.root",
        "${PROJECT_DIR}/root/Feb2021_PH-0750_OM-0p550.root",
        "${PROJECT_DIR}/root/Feb2021_PH-0750_OM-0p950.root",
        # "${PROJECT_DIR}/root/1Million_PH-0125_OM-0p550.root",
        # "${PROJECT_DIR}/root/Feb2021_PH-0150_OM-0p550.root",
        # "${PROJECT_DIR}/root/Feb2021_PH-0150_OM-0p950.root"
    ]

    mass_list = [
        (500,0.48),
        (500,1.1),
        (750,0.55),
        (750,0.95),
        # (125,0.55),
        # (150,0.55),
        # (150,0.95)
    ]

    file_dict = {}
    if len(file_list) != len(mass_list):
        print "list of root files and list of mass points are not the same length:"
        print "file_list: %i\tmass_list: %i"%(len(file_list),len(mass_list))
        quit()
    # validate that all root files exist
    for i in range(len(file_list)):
        f = file_path_check(file_list[i])
        file_dict[f] = mass_list[i]

    moment_list = {}

    # estimate 1st and 2nd gaussian moments for each distribution
    for i,f in enumerate(file_dict.keys()):
        

        # xbins,xlo,xhi,ybins,ylo,yhi = hist_parameters[i]

        hist_2D_i = ROOT.TH2D("hist_2D_i","hist_2D_i",xbins,xlo,xhi,ybins,ylo,yhi)
        hist_2D_i.SetFillColor(22+(i-1)*8)
        hist_2D_i.GetXaxis().SetTitle("PHI")
        hist_2D_i.GetYaxis().SetTitle("OMEGA")
        
        # draw .root data into histogram
        chain = ROOT.TChain(tree_name)
        chain.Add(f)
        chain.Draw(draw_str,cut_str,"goff")

        # remove non-peak bins
        __thresh(hist_2D_i,threshold=0.25)

        # normalize histogram
        hist_2D_i.Scale(1./hist_2D_i.Integral())
        hist_2D_tot.Add(hist_2D_i.Clone())
        
        # ROOT.gStyle.SetPalette(53+5*i)
        hs.Add(hist_2D_i.Clone())

        # fit histogram to 2D gaussian
        xy_gaus_i = ROOT.TF2("xy_gaus_i","xygaus",xlo,xhi,ylo,yhi)
        # xy_gaus_i.SetNpx(xbins)
        # xy_gaus_i.SetNpy(ybins)
        hist_2D_i.Fit("xy_gaus_i","SM")


        # hist_2D_i.Draw("COLZ0"+"SAME")
        # c.Update()

        # temp = raw_input()

        # collect moments from parameters of 2D gaus fit
        moments = {}

        for i in range(1,xy_gaus_i.GetNpar()):
            moments[xy_gaus_i.GetParName(i)] = xy_gaus_i.GetParameter(i)

        # map mass to moments
        phi_mass = file_dict[f][0]
        omega_mass = file_dict[f][1]
        coord_string = "%4.1f,%1.2f"%(phi_mass,omega_mass)
        moment_list[coord_string] = moments

        # xy_gaus_i.Delete()
        hist_2D_i.Delete()

    # plotting
    hist_2D_tot.GetXaxis().SetTitle("PHI")
    hist_2D_tot.GetYaxis().SetTitle("OMEGA")
    hist_2D_tot.SetLineColor(ROOT.kBlack-7)
    hist_2D_tot.SetFillColor(ROOT.kAzure-8)
    # ROOT.gStyle.SetPalette(53+5*i)
    hist_2D_tot.Draw(draw_opt)
    hist_2D_tot.Print()
    c.Update()
    temp = raw_input()

    hs.Draw("LEGO30")
    c.Update()
    temp = raw_input()
    # hist_2D_tot.ShowPeaks(sigma=2,option="",threshold=0.05)
    # c.Update()

    with open(output_file,'w') as json_out:
        dump = json.dumps(moment_list,json_out,indent=4)
        print dump

    temp = raw_input()

def __thresh(hist,threshold=0.05):
        # cut out small bins
        hist_max = hist.GetBinContent(hist.GetMaximumBin())
        for bin_x in range(hist.GetNbinsX()):
            for bin_y in range(hist.GetNbinsY()):
                if hist.GetBinContent(bin_x,bin_y) < threshold*hist_max:
                    hist.SetBinContent(bin_x,bin_y,0.)

def interpolate_moments():
    pass

def main():
    estimate_moments()
    interpolate_moments()

if __name__ == "__main__":
    main()