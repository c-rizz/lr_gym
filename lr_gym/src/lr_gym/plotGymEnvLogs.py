#!/usr/bin/env python3

from cProfile import label
# from copyreg import pickle
from curses import window
from posixpath import abspath
import pickle

from numpy.lib.shape_base import split
from pandas.core.reshape.concat import concat
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse
import time
import signal
import os
import math
import numpy as np
from typing import Dict, List, Tuple

def prepData(csvfiles : List[str],
             x_data_id : str,
             y_data_ids : List[str],
             avgFiles : bool = False,
             xscaling : float = None,
             yscalings : List[float] = None,
             max_x : float = None,
             deparallelize : bool = False,
             avglen : int  = 20,
             parallelsims : int = 1,
             cummax : bool = False,
             nostd : bool = False,
             xoffsets : float = 0.0,
             centeravg : bool = True,
             cutx = [float("-inf"), float("+inf")]):

    in_dfs = [pd.read_csv(csvfile) for csvfile in csvfiles]
    parallel_runs = int(len(in_dfs)/parallelsims)
    print(f"{len(in_dfs)} runs, {parallelsims} parallel sims")

    if max_x is None:
        max_x = float("+inf")
    if xscaling is None:
        xscaling = 1.0

    out_dfs = {}
    file_labels = labelsFromFiles(csvfiles)
    i = 0
    for i in range(len(in_dfs)):
        df = in_dfs[i]
        # df[y_data_ids] = df[y_data_ids]*yscalings
        if "success" in df:
            df["success"].fillna(value=-1,inplace=True)
            df["success"] = df["success"].astype(float) # boolean to 0-1
        df[x_data_id] = df[x_data_id] * xscaling + xoffsets[i]
        df = df.loc[df[x_data_id] < max_x]
        # print(f"cutting {cutx}:\n{df[x_data_id] > cutx[0]}")
        df = df.loc[df[x_data_id] > cutx[0]]
        df = df.loc[df[x_data_id] < cutx[1]]
        # print(f"{df}")
        in_dfs[i] = df
        i+=1

    yid_mean_idx = [yid+"_mean" for yid in y_data_ids]
    yid_min_idx = [yid+"_min" for yid in y_data_ids]
    yid_max_idx = [yid+"_max" for yid in y_data_ids]
    yid_std_idx = [yid+"_std" for yid in y_data_ids]
    yid_means_std_idx = [yid+"_means_std" for yid in y_data_ids]
    yid_means_mean_ciw_idx = [yid+"_means_mean_ciw" for yid in y_data_ids]
    yid_ciw_idx = [yid+"_ciw" for yid in y_data_ids] # confidence interval width
    yid_var_idx = [yid+"_var" for yid in y_data_ids]
    yid_mean_cummax_idx = [yid+"_mean_cummax" for yid in y_data_ids]
    
    if not avgFiles:
        # Compute mean and std for each df
        for i in range(len(in_dfs)):
            df = in_dfs[i]
            rdf = pd.DataFrame()
            rdf[y_data_ids] = df[y_data_ids]
            rdf[yid_mean_idx] = df[y_data_ids].rolling(avglen, center=centeravg).mean()
            rdf[yid_std_idx] = df[y_data_ids].rolling(avglen, center=centeravg).std()
            rdf[yid_ciw_idx] = 1.96*rdf[yid_std_idx]/math.sqrt(avglen)
            rdf[x_data_id] = df[x_data_id]
            rdf[yid_mean_cummax_idx] = rdf[yid_mean_idx].cummax()
            count = rdf[x_data_id].count()
            if count < 30:
                print(f"file {csvfiles[i]} only has {count} samples")
            print(f"{file_labels[i]} has {count} samples")
            out_dfs[file_labels[i]] = rdf
    else:
        for df in in_dfs:
            mx = df[x_data_id].max()
            if mx < max_x:
                max_x = mx
        cut_dfs = []
        for df in in_dfs:
            cut_dfs.append(df[df[x_data_id] < max_x])
        in_dfs = cut_dfs

        if deparallelize:
            raise NotImplementedError()
            i = 0
            for df in in_dfs:
                if x_data_id == "reset_count":
                    df[x_data_id] = df[x_data_id]*parallelsims + i
                    i +=1
                else:
                    df[x_data_id] = df[x_data_id]*parallelsims
            parallelsims = 1
    
        
        per_run_dfs = {}
        for i in range(len(in_dfs)):
            df = in_dfs[i]
            rdf = pd.DataFrame()
            rdf[y_data_ids] = df[y_data_ids]
            rdf[yid_mean_idx] = df[y_data_ids].rolling(avglen, center=centeravg).mean()
            rdf[yid_std_idx] = df[y_data_ids].rolling(avglen, center=centeravg).std()
            rdf[yid_var_idx] = df[y_data_ids].rolling(avglen, center=centeravg).std().pow(2)
            rdf[x_data_id] = df[x_data_id]
            rdf[yid_mean_cummax_idx] = rdf[yid_mean_idx].cummax()
            count = rdf[x_data_id].count()
            if count < 30:
                print(f"file {csvfiles[i]} only has {count} samples")
            print(f"{file_labels[i]} has {count} samples")
            per_run_dfs[file_labels[i]] = rdf
        
        concatdf = pd.concat(per_run_dfs)
        concatdf = concatdf.sort_values(x_data_id)
        mean_df = concatdf.groupby(x_data_id, as_index=False).mean()
        # print(f"concatdf {concatdf.iloc[1000:1010]}")
        means_std_df = concatdf[yid_mean_idx+[x_data_id]].groupby(x_data_id, as_index=False).std()
        # print(f"means_std_df {means_std_df.iloc[125]}\n\n")
        cummax_df = mean_df[yid_mean_cummax_idx]

        if False:   
            in_dfs = [df.sort_values(x_data_id) for df in in_dfs]

            # concatdf = pd.concat([in_dfs], axis=0, keys=list(range(len(in_dfs)))).sort_index(level=1).rolling(3, center=True).sum()
            concatdf = pd.concat(in_dfs, axis=0, keys=list(range(len(in_dfs)))).sort_index(level=0).sort_index(level=1)
            # print(concatdf[x_data_id])    
            rdf = pd.DataFrame()
            rdf[yid_mean_idx] = concatdf[y_data_ids].rolling(window=avglen*len(in_dfs), center=centeravg).mean()
            rdf[yid_min_idx] = concatdf[y_data_ids].rolling(window=avglen*len(in_dfs), center=centeravg).min()
            rdf[yid_max_idx] = concatdf[y_data_ids].rolling(window=avglen*len(in_dfs), center=centeravg).max()
            rdf[yid_std_idx] = concatdf[y_data_ids].rolling(window=avglen*len(in_dfs), center=centeravg).std()
            rdf[yid_ciw_idx] = 1.96*rdf[yid_std_idx]/(math.sqrt(avglen*len(in_dfs)))
            rdf[y_data_ids] = concatdf.groupby(x_data_id)[y_data_ids].mean()
            rdf[x_data_id] = concatdf[x_data_id]
            
            rdf[yid_means_std_idx] = means_std_df[yid_mean_idx] #compute the std between the rolling means of each run
            rdf[yid_means_mean_ciw_idx] = 1.96*means_std_df[yid_mean_idx] /(math.sqrt(len(in_dfs)))
        else:
            rdf = pd.DataFrame()
            rdf[yid_mean_idx] = mean_df[y_data_ids].rolling(window=avglen, center=centeravg).mean()
            # rdf[yid_min_idx] = concatdf[y_data_ids].rolling(window=avglen, center=centeravg).min()
            # rdf[yid_max_idx] = concatdf[y_data_ids].rolling(window=avglen, center=centeravg).max()
            rdf[yid_std_idx] = concatdf.groupby(x_data_id)[yid_mean_idx].std()
            rdf[yid_ciw_idx] = 1.96*rdf[yid_std_idx]/(math.sqrt(len(in_dfs)))
            rdf[y_data_ids] = concatdf.groupby(x_data_id)[y_data_ids].mean()
            # rdf[y_data_ids] = concatdf.groupby(x_data_id)[y_data_ids].mean()
            rdf[x_data_id] = mean_df[x_data_id]
            print(f"concatdf = {concatdf.iloc[800:810]}")
            # print(f"concatdf = {concatdf.groupby(x_data_id)}")
            print(f"mean_df = {mean_df.iloc[100:110]}")
            print(f"rdf = {rdf.iloc[100:110]}")
            
        # print(f"means_std_df[yid_mean_idx] {means_std_df[yid_mean_idx].iloc[125]}")
        # print(f"rdf[yid_means_std_idx] {rdf[yid_means_std_idx].iloc[125]}")
        # print(f"yid_means_mean_ciw_idx {rdf[yid_means_mean_ciw_idx].iloc[125]}")

        # print(rdf)

        # concatdf = pd.concat(in_dfs)
        # concatdf = concatdf.sort_values(x_data_id)
        # by_x = concatdf.groupby(x_data_id)


        # # mean across runs and then window
        # concatdf[yid_mean_idx] = by_x[y_data_ids].mean().rolling(window=avglen, center=True).mean()
        # concatdf[yid_std_idx] = by_x[y_data_ids].std()
        
        # concatdf[yid_std_idx] = concatdf[yid_std_idx].pow(2).rolling(window=avglen, center=True).mean().pow(0.5)
        
        # concatdf.reset_index(drop=True)
        # rdf = pd.DataFrame()
        # rdf[[x_data_id]+y_data_ids+yid_mean_idx+yid_std_idx] = concatdf[[x_data_id]+y_data_ids+yid_mean_idx+yid_std_idx]

        rdf[yid_mean_cummax_idx] = cummax_df
        print(f"maximum mean performance: {rdf[yid_mean_idx].max()} at {rdf[yid_mean_idx].idxmax()}")
        out_dfs["all"] = rdf
        

    for df in out_dfs.values():
        for yids in [y_data_ids , yid_mean_idx , yid_std_idx , yid_mean_cummax_idx]:
            df[yids] = df[yids]*yscalings
    return out_dfs







def makePlot(dfs_dict : Dict[str, pd.DataFrame],
             x_data_id : str,
             max_x : float,
             min_x : float,
             y_data_ids : str,
             max_y : float,
             min_y : float,
             doAvg : bool = False,
             title : str = None,
             xlabel : str = None,
             ylabel : str = None,
             raw : bool = False,
             dfLabels : List[str] = None,
             cummax : bool = False,
             showLegend : bool = True,
             yscalings_dict : Dict[str, float] = None,
             avglen : int = 1,
             runs_number : int = 1,
             palette : List[Tuple[float]] = None,
             legendsize : float = 4,
             minmax = False,
             cutx = [float("-inf"), float("+inf")]):

    for dfk in dfs_dict.keys():
        dfs_dict[dfk] = dfs_dict[dfk].loc[dfs_dict[dfk][x_data_id] > cutx[0]]
        dfs_dict[dfk] = dfs_dict[dfk].loc[dfs_dict[dfk][x_data_id] < cutx[1]]
    plt.clf()

    print(f"{title}")
    print(f"Plotting {len(dfs_dict.keys())} dfs")
    dfs_count = dfs_dict.keys()
    for k, df in dfs_dict.items():
        print(f"{k} cols = "+str(list(df.columns)))
    for k, df in dfs_dict.items():
        df.reset_index(drop = True, inplace=True)
        # print(df.head())
    i = 0
    if dfLabels is None:
        dfLabels = []
        for k, df in dfs_dict.items():
            for yid in y_data_ids:
                dfLabels.append(f"{k}/{yid}×{yscalings_dict[yid]}")
                i+=1



    sns.set_theme(style="ticks") #"darkgrid")
    sns.set_context("paper")

    color_ids = [0]*len(y_data_ids)*len(dfs_count)
    if len(dfs_count) == 1:
        colors_num = len(y_data_ids) # one color per y_data_id
        i = 0
        for file_count in range(len(dfs_count)):
            for yid_count in range(len(y_data_ids)):
                color_ids[i] = yid_count
                i+=1
    else:
        colors_num = len(dfs_count) # one color per file
        i = 0
        for file_count in range(len(dfs_count)):
            for yid_count in range(len(y_data_ids)):
                color_ids[i] = file_count
                i+=1

    # colors_num = len(y_data_ids)*len(dfs_dict.keys())

    # palette = sns.color_palette("tab10")#"husl", len(dfs))
    extended_palette = False
    if palette is None:
        if colors_num < 10 or not extended_palette:
            palette = sns.color_palette("husl", colors_num)
        else:
            palette = sns.color_palette("hls", int((colors_num+1)/2))
            palette += [[e*0.7 for e in c] for c in sns.color_palette("hls", colors_num)]
    # palette[-1] = [0 for e in palette[-1]]
    i = 0
    if raw:
        for k, df in dfs_dict.items():
            for yid in y_data_ids:
                c = palette[color_ids[i]]
                if doAvg:
                    c = [(e+1)/2 for e in c]
                p = sns.lineplot(data=df,x=x_data_id,y=yid,color=c, alpha = 0.7, ci=None, linewidth=0.5) #, ax = ax) #
                i+=1
    i = 0
    if doAvg:                                
        for k, df in dfs_dict.items():
            for yid in y_data_ids:
                c = palette[color_ids[i]]
                l = dfLabels[i]
                print(f"Plotting {k}/{yid} mean, {len(df[x_data_id])} samples, max_avg = {df[yid+'_mean'].max()}, min_avg = {df[yid+'_mean'].min()}, max = {df[yid].max()}, min = {df[yid].min()}, last_avg = {df[yid+'_mean'].iloc[-1]}, last = {df[yid].iloc[-1]}")
                p = sns.lineplot(x=df[x_data_id],y=df[yid+"_mean"], color=c, label=l, ci=None, linewidth=0.5) #, ax = ax) #
                if not raw and yid+"_std" in df and not minmax:
                    print(f"Plotting {k}/{yid} std")
                    ci_widths = df[yid+"_ciw"] #1.96*df[yid+"_std"]/(math.sqrt(avglen*runs_number))
                    cis = (df[yid+"_mean"] - ci_widths, df[yid+"_mean"] + ci_widths)
                    c = [(e+1)/2 for e in c]
                    p.fill_between(df[x_data_id],cis[0],cis[1], color=c, alpha = 0.5)
                if minmax:
                    c = [(e+1)/2 for e in c]
                    p.fill_between(df[x_data_id],df[yid+"_min"],df[yid+"_max"], color=c, alpha = 0.5)
                i+=1
    i = 0
    if cummax:
        for k, df in dfs_dict.items():
            for yid in y_data_ids:
                c = palette[color_ids[i]]
                c = [a*0.75 for a in c]
                l = dfLabels[i]
                print(f"plotting cummax {df[yid+'_mean_cummax']}")
                p = sns.lineplot(x=df[x_data_id],y=df[yid+"_mean_cummax"], color=c, label=l, ci=None, linewidth=0.5) #, ax = ax) #
                i+=1
                
    #plt.legend(loc='lower right', labels=names)
    # pathSplitted = os.path.dirname(csvfile).split("/")
    # plt.title(pathSplitted[-2]+"/"+pathSplitted[-1]+"/"+os.path.basename(csvfile))

    p.set_xlim(min_x,max_x) # If None they leave the current limit


    if max_y is not None or min_y is not None:
        p.set_ylim(min_y,max_y)


    if xlabel is not None:
        p.set_xlabel(xlabel)
    if ylabel is not None:
        p.set_ylabel(ylabel)

    if showLegend:
        # p.legend(loc='upper left')
        p.legend(prop={'size': legendsize}) #, loc='lower right')
    else:
        p.legend().remove()
    p.minorticks_on()
    #p.tick_params(axis='x', which='minor', bottom=False)
    p.grid(linestyle='dotted',which="both")

    p.set_aspect(1.0/p.get_data_ratio()*0.5)

    if title is not None:
        plt.title(title)
    plt.tight_layout()



ctrl_c_received = False
def signal_handler(sig, frame):
    #print('You pressed Ctrl+C!')
    global ctrl_c_received
    ctrl_c_received = True

def labelsFromFiles(csvfiles : List[str]):
    commonPath = os.path.commonpath([os.path.abspath(os.path.dirname(cf)) for cf in csvfiles])
    commonRealPath = os.path.realpath(commonPath) # absolute path without links
    lastCommonPathElement = commonRealPath.split("/")[-1]
    dfLabels = [None]*len(csvfiles)
    i = 0
    for csvfile in csvfiles:
        dfLabels[i] = ""
        splitpath = os.path.realpath(csvfile).split("/")
        dfLabels[i] = splitpath[splitpath.index(lastCommonPathElement)+1]
        for f in splitpath:
            if f.startswith("seed_"):
                dfLabels[i] += "/"+f[5:]
        if dfLabels[i] is None:
            dfLabels[i] = chr(65+i)
        i+=1
    has_duplicates = len(dfLabels) != len(set(dfLabels))
    if has_duplicates:
        i = 0
        for csvfile in csvfiles:
            split = csvfile.split("/")
            for j in range(len(split)-1):
                if split[j+1].startswith("seed_"):
                    dfLabels[i] = split[j]+"/"+dfLabels[i]
            i+=1
    return dfLabels
            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvfiles", nargs="+", required=False, type=str, help="Csv file(s) to read from")
    ap.add_argument("--nogui", default=True, action='store_true', help="Dont show the plot window, just save to file")
    ap.add_argument("--noavg", default=False, action='store_true', help="Do not plot curve average")
    ap.add_argument("--avgfiles", default=False, action='store_true', help="Make an average pof the provided files instead of displaying all of them")
    ap.add_argument("--raw", default=False, action='store_true', help="Plot raw data")
    ap.add_argument("--maxx", required=False, default=None, type=float, help="Maximum x value to plot")
    ap.add_argument("--minx", required=False, default=None, type=float, help="Minimum x value to plot")
    ap.add_argument("--maxy", required=False, default=None, type=float, help="Maximum y axis value")
    ap.add_argument("--miny", required=False, default=None, type=float, help="Minimum y axis value")
    ap.add_argument("--period", required=False, default=-1, type=float, help="Seconds to wait between plot update")
    ap.add_argument("--out", required=False, default=None, type=str, help="Filename for the output plot")
    ap.add_argument("--ydataids", nargs="+", required=False, default=None, type=str, help="Ids for y values")
    ap.add_argument("--yscalings", nargs="+", required=False, default=None, type=float, help="Scaling factors for y values")
    ap.add_argument("--xdataid", required=False, default="reset_count", type=str, help="Data to put on the x axis")
    ap.add_argument("--avglen", required=False, default=20, type=int, help="Window size of running average")
    ap.add_argument("--xscaling", required=False, default=1, type=float, help="Scale the x values by this factor")
    ap.add_argument("--xlabel", required=False, default=None, type=str, help="label to put on x axis")
    ap.add_argument("--ylabel", required=False, default=None, type=str, help="label to put on y axis")
    ap.add_argument("--title", required=False, default=None, type=str, help="plot title")
    ap.add_argument("--format", required=False, default="pdf", type=str, help="format of the output file")
    ap.add_argument("--savedfs", default=False, action='store_true', help="Save prepped dataframes as csv")
    ap.add_argument("--dontplot", default=False, action='store_true', help="Do not plot")
    ap.add_argument("--pklfiles", nargs="+", required=False, default=None, type=str, help="load already prepared pkl files")
    ap.add_argument("--deparallelize", default=False, action='store_true', help="Transform data collected in parallel in sequential data")
    ap.add_argument("--parallelsims", required=False, default=1, type=int, help="Number of parallel simulators per run")
    ap.add_argument("--legend", nargs="+", required=False, default=None, type=str, help="List of the labels to put in the legend")
    ap.add_argument("--nolegend", default=False, action='store_true', help="Hide legend")
    ap.add_argument("--cummax", default=False, action='store_true', help="Plot cumulative maximum")
    ap.add_argument("--nostd", default=False, action='store_true', help="Dont plot std")
    ap.add_argument("--outfname", required=False, default=None, type=str, help="Name of the output file (without path)")
    ap.add_argument("--reorderfiles", nargs="+", required=False, default=None, type=int, help="Reorder the input files before processing")
    ap.add_argument("-p","--palette", nargs="+", required=False, default=None, action='append', type=float, help="palette to use as -p <r> <g> <b> -p <r> <g> <b> ...")
    ap.add_argument("--xoffsets", nargs="+", required=False, default=[0], type=int, help="Add this to x data")
    ap.add_argument("--nocenteravg", default=False, action='store_true', help="Do not center the window averaging")
    ap.add_argument("--legendsize", required=False, default=4, type=float, help="Size of the legend")
    ap.add_argument("--cutx", nargs=2, required=False, default=[float("-inf"),float("+inf")], type=float, help="Cut samples to be in this x interval")


    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    if not args["nogui"]:
        signal.signal(signal.SIGINT, signal_handler)

    print(args["legend"])

    matplotlib.rcParams['figure.raise_window'] = False
    #matplotlib.use('Tkagg')
    if not args["nogui"]:
        plt.ion()
        plt.show()

    if args["ydataids"] is not None:
        y_data_ids = args["ydataids"]
    else:
        y_data_ids=["ep_reward"]

    yscalings = args["yscalings"] if args["yscalings"] is not None else [1.0]*len(y_data_ids)


    #fig, ax = plt.subplots(figsize=(11, 8.5))
    while not ctrl_c_received:
        #print("Plotting")
        try:
            csvfiles = args["csvfiles"]

            if args["reorderfiles"] is not None:
                csvfiles = [csvfiles[i] for i in args["reorderfiles"]]
                
            if args["avgfiles"]:
                runs_num = len(csvfiles)
            else:
                runs_num = None
            commonPath = os.path.commonpath([os.path.abspath(os.path.dirname(cf)) for cf in csvfiles])
            commonRealPath = os.path.realpath(commonPath) # absolute path without links
            if args["out"] is not None:
                out_path = args["out"]
            else:
                if args["outfname"] is None:
                    fname = "_".join(y_data_ids)
                else:
                    fname = args["outfname"]
                if args["avgfiles"]:
                    fname+="_avg"
                out_path = commonPath+"/"+fname+"."+args["format"]
            title = args["title"]
            if title is None:
                crps = commonRealPath.split("/")
                if crps[-1] == "eval":
                    title = crps[-3]+"/"+crps[-2]+"/"+crps[-1]
                else:
                    title = crps[-2]+"/"+crps[-1]
            if title.lower() == "none":
                title = None
            
            if args["pklfiles"] is not None:
                dfs_byfile = {f : pickle.load(open(f, "rb" )) for f in args["pklfiles"]}
                dfs = {}
                for file, fdfs in dfs_byfile.items():
                    for yid, df in fdfs.items():
                        dfs[file+"/"+yid] = df
                # dfs = {f : pickle.load(open(f, "rb" )) for f in args["pklfiles"]}

            else:
                if len(args["xoffsets"])==1:
                    args["xoffsets"] = args["xoffsets"] * len(args["csvfiles"])
                dfs = prepData(csvfiles=csvfiles,
                                x_data_id=args["xdataid"],
                                y_data_ids=y_data_ids,
                                avgFiles=args["avgfiles"],
                                xscaling=args["xscaling"],
                                yscalings=yscalings,
                                max_x = args["maxx"],
                                deparallelize = args["deparallelize"],
                                avglen=args["avglen"],
                                cummax = args["cummax"],
                                parallelsims= args["parallelsims"],
                                nostd=args["nostd"],
                                xoffsets = args["xoffsets"],
                                centeravg=not args["nocenteravg"],
                                cutx = args["cutx"])
            if args["savedfs"]:                
                pickle.dump(dfs,open(out_path+".pkl", "wb" ))
            if not args["dontplot"]:
                if args["legend"] is not None:
                    dfLabels = args["legend"]
                elif args["avgfiles"]:
                    dfLabels = None
                else:
                    dfLabels = None # labelsFromFiles(csvfiles)

                yscalings_dict = {y_data_ids[i] : yscalings[i] for i in range(len(y_data_ids))}
                makePlot(dfs,
                        x_data_id=args["xdataid"],
                        max_x = args["maxx"],
                        min_x = args["minx"],
                        y_data_ids=y_data_ids,
                        max_y = args["maxy"],
                        min_y= args["miny"],
                        doAvg = not args["noavg"],
                        title = title,
                        xlabel = args["xlabel"],
                        ylabel = args["ylabel"],
                        raw = args["raw"],
                        dfLabels=dfLabels,
                        cummax=args["cummax"],
                        showLegend=not args["nolegend"],
                        yscalings_dict=yscalings_dict,
                        avglen = args["avglen"],
                        runs_number=runs_num,
                        palette = args["palette"],
                        legendsize = args["legendsize"],
                        cutx = args["cutx"])
                if args["format"] == "png":
                    plt.savefig(out_path, dpi=600,bbox_inches='tight')
                else:
                    plt.savefig(out_path,bbox_inches='tight')
                print("Saved to "+os.path.realpath(out_path))

                #plt.show(block=True)
                if not args["nogui"]:
                    plt.draw()
                    plt.pause(0.01)
                    if args["once"]:
                        plt.show(block=True)
                        break
        except pd.errors.EmptyDataError:
            print("No data...")
        except FileNotFoundError as e:
            print("File not present... e="+str(e))
        if args["period"] <= 0:
            break
        plt.pause(args["period"])

if __name__ == "__main__":
    main()