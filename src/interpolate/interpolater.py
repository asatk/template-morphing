import ROOT
import time
import random
import json
import os
import math
import re

from collections import deque

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

from build.fitter import fitter

ROOTCOLORS = [
    kPink+1,
    kOrange+1,
    kYellow,
    kGreen-9,
    kRed,
    kGreen,
    kBlue,
    kCyan,
    kMagenta,
    kYellow,
    kGray,
    kBlack,
    kOrange,
    kSpring,
    kTeal,
    kAzure,
    kViolet,
    kPink]

ROOTCOLORS2 = [
    kRed,
    kGreen,
    kBlue-3,
    kViolet,
    kOrange
]

class interpolater:
    def __init__(self,fit_info_list,q=deque()):
        self.q = q

        #set up fitter
        self.ftr = fitter(fit_info_list[0])

        # self.__funcs(fit_info_list)

        # self.fit_info_list = fit_info_list

        #set up dictionary between mass points and fits
        self.__mass_pts(fit_info_list)

        #set drawing parameters
        self.drawing_params = {
            'color':kOrange+1,
            'pdf_color':kBlack,
            'cdf_color':kBlack
        }

        #set up canvas
        self.canv = ROOT.TCanvas("canv","interpolater analysis",1200,900)
        self.canv.DrawCrosshair()

    def interpolate(self):
        self.canv.cd()

        hasStack = False
        hasPoint = False
        Npx = self.ftr.bins
        res = 20
        method = ""
        
        interp_cdflist = []
        interp_flist = []
        interp_append = False
        cmd = " "
        while not cmd == "":
            if len(self.q) > 0:
                cmd = self.q.pop()
            else:
                print ['+','c','c.pdf','c.cdf','npx','pt','i-pdf','i-cdf','pdf','cdf','n']
                cmd = raw_input("cmd: ")

            if cmd == 'i':
                print "[INTERPOLATING WITH SPECIFIED METHOD]"
                
                if method == "hist":
                    self.__interp_hist()
                elif method == "param":
                    self.__interp_parameter(ptcl1,ptcl1_mass,ptcl2_mass)
                else:
                    print "INTERPOLATION METHOD NOT VALID - Choose from: [HIST, PARAM]"
            elif cmd == 'l':
                print "[LOADING TOOL WITH PARAMETERS]"
                ptcl1 = raw_input("Particle 1 Name (phi, omega): ").lower()
                ptcl1_mass = float(raw_input("%s Mass (float): "%(ptcl1)))
                ptcl2_mass = float(raw_input("%s Mass (float): "%(list(set(['phi','omega'])-set([ptcl1]))[0])))
                method = raw_input("Interpolation Method (hist, param): ").lower()
                formula = raw_input("Interpolation Function (gaus, crystalball, landau, landxgaus): ")
                run = raw_input("Which Run (Jun2020, Feb2021, Mar2021, 1Million, ALL): ")
                self.__funcs(ptcl1,ptcl1_mass,ptcl2_mass,formula,run)






            elif cmd == 'c':
                print "c - [PICK COLOR]"
                if len(q) > 0:
                    color = eval(q.pop())
                else:
                    color = eval(raw_input("c - pick your color (0-40) or kColor+-10: "))
            elif cmd == 'c.pdf':
                print "c.pdf - [COLORING ALL PDFS]"
                if len(q) > 0:
                    pdf_color = eval(q.pop())
                else:
                    pdf_color = eval(raw_input("c.pdf - pick your color (0-40) or kColor+-10: "))
            elif cmd == 'c.cdf':
                print "c.cdf - [COLORING ALL CDFS]"
                if len(q) > 0:
                    cdf_color = eval(q.pop())
                else:
                    cdf_color = eval(raw_input("c.cdf - pick your color (0-40) or kColor+-10: "))
            elif cmd == 'npx':
                print "pt - [SPECIFY NUMBER OF POINTS IN FUNCTION]"
                if len(q) > 0:
                    npx_input = q.pop()
                else:
                    npx_input = raw_input("npx - set Npx to: ")
                Npx = int(npx_input) if (npx_input != "" and int(npx_input) <= 10000) else ftr.bins
                print "npx - Npx set to",Npx
            elif cmd == 'pt':
                print "pt - [SPECIFYING POINT OF INTERPOLATION]"
                if len(q) > 2:
                    point = float(q.pop())
                    axis = bool(q.pop())
                    res = int(q.pop())
                    interp_method = q.pop()
                else:
                    point = float(raw_input("pt - data point for %s: "%(ftr.var)))
                    axis = float(raw_input("pt - axis of interpolation [MAJOR (True) or MINOR (False)]:"))
                    res_str = raw_input("pt - data resolution (num pts): ")
                    if res_str != "":
                        res = int(res_str)
                    interp_method_str = raw_input("pt - choose interpolation method: ['HIST','PARAM']: ")
                    if interp_method_str != "":
                        interp_method = interp_method_str
                hasPoint = True
                print "pt - interpolating at mass point %4.3f %s with %i points"%(point,"GeV" if ftr.pname == 'omega' else "MeV",res)
            elif cmd == 'i-pdf':
                if 'interp_method' not in locals():
                    print "i-pdf - must build cdfs from interpolating methods available [i-cdf]"
                else:
                    print "i-pdf - [INTERPOLATING PDF]"
                    if interp_method == "HIST":
                        print interp_flist
                        print interp_cdflist
                        cdf_derivative(point,interp_flist,interp_cdflist,self.canv,color=color)
                    elif interp_method == "PARAM":
                        for n,interp_f in enumerate(interp_flist):
                            for p in range(interp_f.GetNpar()):
                                print "i-pdf[PARAM] - interp_f#%i param %i %4.3f"%(n,p,interp_f.GetParameter(p))
                            # interp_f.SetParameter(3,1.4)
                            # interp_f.SetParameter(4,5.)
                    first = True
                    for interp_f in interp_flist:
                        interp_f.Draw(
                                "L" + 
                                ("SAME" if not first else "") +
                                ("A" if interp_method == "HIST" else ""))
                        self.canv.Update()
                        first = False
            elif cmd == 'i-cdf':
                if not hasPoint:
                    print "i-cdf - need point: use 'pt' to specify"
                elif not hasStack:
                    print "i-cdf - need histograms/functions: use 'pdf or 'cdf' to generate"
                else:
                    print "i-cdf - [INTERPOLATING CDF - %s]"%(interp_method)
                    if interp_method == "HIST":
                        c = ROOT.TCanvas("name","temp",600,600)
                        thing = interp_hist(fit_info_list,res,point,axis,interp_cdflist,self.canv,c,ftr=ftr,cdf_color=cdf_color)
                        print thing
                        print "post"
                    elif interp_method == "PARAM":
                        interp_parameter(flist,masspts,point,axis,interp_flist,self.canv,ftr=ftr,Npx=Npx)
                        print "post"
                        # first = True
                        # for f in interp_flist:
                        #     f.Draw("C" + "SAME" if not first else "")
                        #     self.canv.Update()
                        #     f.Print()
                        #     for p in range(f.GetNpar()):
                        #         print f.GetParameter(p)
            elif cmd == 'pdf':
                print "pdf - [PDFs]"
                hstack = ROOT.THStack("hs","%s of %s for %s"
                        %("Probability Distributions" if normalized else "Event Distributions",ftr.var,ftr.pname))
                flist = ROOT.TList()
                for count,i in enumerate(fit_info_list):
                    # set up fitter and fit info
                    print "using fit model #%i: %s"%(count,i)
                    json_file = open(i,'r')
                    info = json.load(json_file)
                    run_file = info['run_file']

                    ftr = fitter(run_file,fitted=True,fit_info=i)
                    if pdf_color is None:
                        ftr.func.SetLineColor(ROOTCOLORS[count])
                    else:
                        ftr.func.SetLineColor(pdf_color)
                    ftr.func.SetLineWidth(5)
                    ftr.func.SetNpx(Npx)
                    func = ftr.func.Clone()
                    flist.Add(func)
                    fhist = ROOT.TH1D(func.GetHistogram())
                    hstack.Add(fhist)
                    json_file.close()

                print hstack.GetNhists()
                hstack.GetHists().Print()
                hstack.Draw("hist nostack")
                self.canv.Update()
                for interp_f in interp_flist:
                    interp_f.Draw("C SAME")
                    self.canv.Update()
                hasStack = True
            elif cmd == 'cdf':
                print "cdf - [CDFs]"
                hstack = ROOT.THStack("hs","%s of %s for %s"
                        %("Cumulative Distributions" if normalized else "Event Counts",ftr.var,ftr.pname))
                flist = ROOT.TList()
                
                for count,i in enumerate(fit_info_list):
                    # set up fitter and fit info
                    print "using fit model #%i: %s"%(count,i)
                    json_file = open(i,'r')
                    info = json.load(json_file)
                    run_file = info['run_file']

                    ftr = fitter(run_file,fitted=True,fit_info=i)
                    if cdf_color is None:
                        ftr.func.SetLineColor(ROOTCOLORS[count])
                    else:
                        ftr.func.SetLineColor(cdf_color)
                    ftr.func.SetLineWidth(5)
                    ftr.func.SetNpx(Npx)
                    func = ftr.func.Clone()
                    flist.Add(func)
                    fhist = ROOT.TH1D(func.GetHistogram().DrawNormalized().GetCumulative())
                    hstack.Add(fhist)
                    json_file.close()
                
                self.canv.Clear()
                first = True
                hstack.Draw("hist nostack")
                self.canv.Update()
                hasStack = True
            elif cmd == 'n':
                if not normalized:
                    print 'n - [NORMALIZING]'
                    for count,name in enumerate(fit_info_list):
                        idx = name.rfind('/')
                        fit_info_list[count] = name[:idx+1] + "norm" + name[idx+1:]
                    normalized = True
                else:
                    print 'n - [DENORMALIZING]'
                    for count,name in enumerate(fit_info_list):
                        idx = name.rfind('/')
                        fit_info_list[count] = name[:idx+1] + name[idx+5:]
                    normalized = False

    # def __funcs(self,fit_info_list):
    #     self.__func_list = []
    #     for f in fit_info_list:
    #         with json.open(f) as json_file:
    #             info = json.load(json_file)
    #             temp_func = ROOT.TF1(f,)

    def __funcs(self,ptcl1,ptcl1_mass,ptcl2_mass,formula,run):
        self.__func_list = []
        directory = r"../out/%s/%s/"%(ptcl,formula)
        if ptcl1 == 'phi':
            ph_mass = ptcl1_mass
            om_mass = ptcl2_mass
        else:
            ph_mass = ptcl2_mass
            om_mass = ptcl1_mass
        # p = r"%s_PH-%04i_OM-%sp%s\.root"%(run,int(ph_mass),str("{:.3f}".format(om_mass)[0],str("{:.3f}".format(om_mass)[2:]))
        # dir_list = list([file for file in os.listdir(directory) if re.search(p,file) is not None])




    def __mass_pts(self,fit_info_list):
        #get list of mass points
        self.ph_mass_pts = dict()
        self.om_mass_pts = dict()
        for f in fit_info_list:
            p = r"\w+_PH-(\d{4})_OM-(\d)p(\d{3}).json$"
            reg = re.search(p,f)
            
            ph_mass = float(reg.group(1))
            if ph_mass not in self.ph_mass_pts:
                self.ph_mass_pts[ph_mass] = set([f])
            else:
                self.ph_mass_pts[ph_mass].add(f)
            
            om_mass = float(reg.group(2)+"."+reg.group(3)) 
            if om_mass not in self.om_mass_pts:
                self.om_mass_pts[om_mass] = set([f])
            else:
                self.om_mass_pts[om_mass].add(f)

        print "*ph",self.ph_mass_pts
        print "*ph keys",list(self.ph_mass_pts)
        print "*om",self.om_mass_pts
        print "*om keys",list(self.om_mass_pts)

    def __interp_hist(self,fit_info_list,res,point,axis,interp_cdflist,c,**kwargs):
        cdflist = ROOT.TList()
        # self.canv.Clear()
        # c = ROOT.TCanvas()
        ftr = kwargs['ftr']
        cdf_color = kwargs['cdf_color']
        fit_name = ftr.fit_name
        var = ftr.var
        pname = ftr.pname
        # chain = ROOT.TChain("twoprongNtuplizer/fTree")

        hstack = ROOT.THStack("hs","%s of %s for %s"
                %("Cumulative Distributions",ftr.var,ftr.pname))

        c.cd()
        first = True
        for num,f in enumerate(fit_info_list):
            with open(f) as json_file:
                info = json.load(json_file)

                exec('hist'+str(num)+' = ROOT.TH1D("hist"+str(num),"%s cdf - %s"%(ftr.fit_name,ftr.var),ftr.bins,ftr.lo,ftr.hi)')

                hist = eval('hist'+str(num))

                x_ax = hist.GetXaxis()
                if ftr.pname == 'phi':
                    x_ax.SetTitle("%s (MeV)"%(ftr.var))
                else:
                    x_ax.SetTitle("%s (GeV)"%(ftr.var))
                x_ax.CenterTitle(True)
                y_ax = hist.GetYaxis()
                y_ax.SetTitle("Event Probability Density")
                y_ax.CenterTitle(True)
                hist.SetMaximum(250)

                if cdf_color is None:
                    hist.SetLineColor(ROOTCOLORS[num])
                    hist.SetLineColor(ROOTCOLORS[num]+1)
                else:
                    hist.SetLineColor(cdf_color)
                    hist.SetLineColor(cdf_color+1)

                hist.SetLineWidth(3)
                hist.SetStats(0)

                chain = ROOT.TChain("twoprongNtuplizer/fTree")
                chain.Add(info['run_file'])
                draw_s = info['var'] + ">>hist"+str(num)
                cut_s = info['cuts']
                chain.Draw(draw_s, cut_s, "goff")
                # chain.Draw(draw_s, cut_s, option = "goff") # no keyword arg support?
                hist_cum = hist.DrawNormalized().GetCumulative()
                hist_cum.SetMaximum(1.05)
                hstack.Add(hist_cum.Clone())
                c.Update()
                first = False
                json_file.close()
        c.Clear()
        hstack.Print()
        hstack.Draw("hist nostack")
        c.Update()

        inv_graph_list = []
        idx = 1
        if (hstack.GetNhists() > 2):
            for jdx,h in enumerate(hstack.GetHists()):
                g = ROOT.TGraph(h)
                xpts = g.GetY()
                ypts = g.GetX()
                inv_graph_list.append(ROOT.TGraph(ftr.bins,xpts,ypts))
                if point < inv_graph_list[jdx].Eval(0.5):
                    idx = jdx
                    inv_graph_list = inv_graph_list[idx-1:idx+1]
                    break

        pct = abs((point - inv_graph_list[0].Eval(0.5))/(inv_graph_list[0].Eval(0.5) - inv_graph_list[1].Eval(0.5)))
        print "new percent calc w/ TGraph of inv w lin interp by TSpline: pct - ",pct

        color = None
        outerfirst = True
        innerfirst = True
        interp_cdf = ROOT.TGraph()
        interp_cdf.SetLineColor(color if color is not None else ROOTCOLORS[6])
        interp_cdf.SetLineWidth(5)
        interp_cdf.SetNameTitle("interp_cdf_"+str(len(interp_cdflist)),"morphed cdf for %4.3f"%(point))
        interp_cdf.SetPoint(0,0.,0.)

        # get x values for both interpolating cdfs from equal y-values at some step determined by Npx
        for y in [float(i) / res for i in range(res+1)]:
            print "i-cdf[HIST] - step: %i ; level: y=%4.3f"%(int(round(y*res)),y)
            pts = []
            for num,cdf in enumerate(inv_graph_list):
                x = cdf.Eval(y)
                # print x,y
                # if math.isnan(x):
                #     x = 0
                pts.append(x)
                print "i-cdf[HIST] - cdf_%i @ %1.3f = %4.3f (%s %s)"%(
                        (num+idx-1)%hstack.GetNhists(),y,x,var,"MeV" if pname == 'phi' else "GeV")
                innerfirst = False
            # print pts
            # print pct
            interp_x = pts[0] + pct * (pts[1] - pts[0])
            interp_cdf.SetPoint(int(round(y*res)+1),interp_x,y)

            outerfirst = False
        
        #point removal process, look into later
        # remove_pts = []
        # if math.isnan(interp_cdf.GetPointX(0)):
        # if math.isnan(interp_cdf.GetX()[0]):
        #     interp_cdf.SetPoint(0,0.,0.)
        # for i in range(res+1):
        #     # p = interp_cdf.GetPointX(i)
        #     p = interp_cdf.GetX()[i]
        #     # print i,p
        #     if i != 0 and (math.isnan(p) or p <= 0):
        #         remove_pts.append(i)

        # remove_count = 0
        # for r_n,r_pt in enumerate(remove_pts):
        #     print "removed",r_pt
        #     interp_cdf.RemovePoint(r_pt-r_n)
        
        interp_cdf.SetPoint(interp_cdf.GetN(),ftr.hi,1.0)
        interp_cdf.Draw("SAME")
        c.Update()
        interp_cdflist.append(interp_cdf)
        # interp_cdf.Print()
        
        # hist_list[idx-1].Print()
        # hist_list[idx].Print()
        # interp_cdf.Print()
        self.canv.cd()
        # time.sleep(10)
        return 1

    def __interp_parameter(self,ptcl1,ptcl1_mass,ptcl2_mass,formula,run,**kwargs):
        print "[INTERPOLATING PARAMETERS WITH OLS]"
        
        # for f in 


        param_flist = ROOT.TList()
        param_masspts = []

        
        
        # param_flist.Print()
        # print param_masspts

        param_lines = []
        interp_params = []
        mg = ROOT.TMultiGraph()
        ftr = kwargs['ftr']
        Npx = kwargs['Npx']
        # iterate through parameters - make one fit per parameter for all functions
        for i in range(param_flist[0].GetNpar()):
            g = ROOT.TGraph()
            g.SetLineWidth(5)
            g.SetLineColor(ROOTCOLORS[i+5])
            g.SetNameTitle("OLS - %s#%i"%(param_flist[0].GetParName(i),i))

            par_names = list(param_flist[j].GetParName(i) for j in range(2))
            par_values = list(param_flist[j].GetParameter(i) for j in range(2))
            pars = zip(par_names,par_values)
            print "i-cdf[PARAM] - ",pars

            pts = {}
            param_vals = []
            # gather each fn's parameter value for current parameter
            for num,f in enumerate(param_flist):
                print "i-cdf[PARAM] - %s#%i: %s = %6.4f"%(f.GetName(),num,f.GetParName(i),f.GetParameter(i))
                param_vals.append(f.GetParameter(i))
                pts[num] = f.GetParameter(i)
                g.SetPoint(num,param_masspts[num],f.GetParameter(i))
            g.Print()
            
            param_line = ROOT.TF1("Line#%i"%(i),"[m]*x[0]+[b]",ftr.lo,ftr.hi)
            param_line.SetLineColor(kRed)
            param_line.SetLineWidth(3)
            line_m = (param_vals[1] - param_vals[0]) / (param_masspts[1] - param_masspts[0])
            line_b = param_vals[0] - line_m * param_masspts[0]
            # g.Fit(param_line,"SM0Q rob=0.8")
            # param_OLS = g.GetFunction("OLS#%i"%(i))
            param_line.SetParameter("m",line_m)
            param_line.SetParameter("b",line_b)
            param_lines.append(param_line)
            interp_params.append(param_line.Eval(point))
            # param_lines.append(param_OLS)
            # interp_params.append(param_OLS.Eval(point))
            mg.Add(g)

        mg.Draw("AL")
        self.canv.Update()
        interp_f = ROOT.TF1("param interp",str(param_flist[0].GetExpFormula()),ftr.lo,ftr.hi)
        interp_f.SetLineWidth(5)

        print "i-cdf[PARAM] - formula",interp_f.GetExpFormula()
        for j in range(interp_f.GetNpar()):
            print "i-cdf[PARAM] - param",j,param_flist[0].GetParName(j),interp_params[j]
            interp_f.SetParameter(param_flist[0].GetParName(j),interp_params[j])
            param_lines[j].Draw("SAME")
            self.canv.Update()

        interp_f.SetNpx(ftr.bins)
        print "i-cdf[PARAM] - interp_f integral",interp_f.GetHistogram().Integral()
        print "i-cdf[PARAM] - norm factor",(1. / interp_f.GetHistogram().Integral())
        interp_f.SetParameter('Constant',interp_f.GetParameter('Constant') * 1. / interp_f.GetHistogram().Integral())
        print "i-cdf[PARAM] - integral - should be 1.0:",interp_f.GetHistogram().Integral()

        interp_flist.append(interp_f)
        ROOT.gDirectory.Append(mg)

    def cdf_derivative(self,point,interp_flist,interp_cdflist,**kwargs):
        color = kwargs['color']
        # self.canv.Clear()
        interp_mg = ROOT.TMultiGraph()
        num = 0
        interp_cdf = interp_cdflist[0]
        # for num,interp_cdf in enumerate(interp_cdflist):
        interp_f = ROOT.TGraph()
        interp_f.SetNameTitle("interp_f_"+str(num),"morphed func for %4.3f"%(point))
        interp_f.SetLineColor(color if color is not None else ROOTCOLORS[num])
        interp_f.SetLineWidth(5)
        for i in range(interp_cdf.GetN()-1):
            dydx = (interp_cdf.GetPointY(i+1)-interp_cdf.GetPointY(i))/(interp_cdf.GetPointX(i+1)-interp_cdf.GetPointX(i))
            print "i-pdf[HIST] - dydx[%i]=%2.4f"%(i,dydx)
            interp_f.SetPoint(i,interp_cdf.GetPointX(i),dydx)
        interp_f.SetPoint(interp_cdf.GetN()-1,interp_cdf.GetPointX(interp_cdf.GetN()-1),0.)
        print "INTERP_F#"+str(num)
        interp_f.Print()
        print "INTERP EVAL1a",interp_f.Eval(interp_f.GetPointX(1)/2)
        print "INTERP EVAL1b",interp_f.Eval(interp_f.GetPointX(1)/2,0,"")
        print "INTERP EVAL2",interp_f.Eval(interp_f.GetPointX(1)/2,0,"S")
        temp_res = 200
        end_pt = interp_f.GetPointX(1)
        for i in range(temp_res):
            pt = i*end_pt/temp_res
            print "~~~PT - %4.2f~~~"%(pt)
            print "INTERP EVAL1b",interp_f.Eval(pt,0,"")
            print "INTERP EVAL2",interp_f.Eval(pt,0,"S")
        # print interp_f.Integral(0,interp_cdf.GetN())
        interp_flist.append(interp_f)
        # interp_flist.Add(interp_f)
        # interp_mg.Add(interp_f)

        # interp_mg.Draw("AC")
        return interp_flist

if __name__ == "__main__":
    print sys.argv
    # interpolate_fit()