#!/usr/bin/env python3

from posixpath import abspath

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
from typing import List

def prepData(csvfiles : List[str],
             x_data_id : str,
             avgFiles : bool = False,
             xscaling : float = None,
             max_x : float = None,
             deparallelize : bool = False,
             avglen : int  = 20,
             parallelsims : int = 1,
             cummax : bool = False,
             nostd : bool = False):

    in_dfs = [pd.read_csv(csvfile) for csvfile in csvfiles]
    parallel_runs = int(len(in_dfs)/parallelsims)
    print(f"{len(in_dfs)} runs, {parallelsims} parallel sims")
    out_dfs = []
    for df in in_dfs:
        if "success" in df:
            df["success"].fillna(value=-1,inplace=True)
            df["success"] = df["success"].astype(int) # boolean to 0-1
        if xscaling is not None:
            df[x_data_id] = df[x_data_id] * xscaling
        if max_x is not None:
            df = df.loc[df[x_data_id] < max_x]
    if not avgFiles:
        for df in in_dfs:
            rdf = pd.DataFrame()
            rdf[y_data_id] = df[y_data_id]
            rdf["mean"] = df[y_data_id].rolling(avglen, center=True).mean()
            rdf["std"] = df[y_data_id].rolling(avglen, center=True).std()
            rdf[x_data_id] = df[x_data_id]
            out_dfs.append(rdf)
    else:
        if max_x is None:
            max_x = float("+inf")
            for df in in_dfs:
                mx = df[x_data_id].max()
                if mx < max_x:
                    max_x = mx
        if deparallelize:
            i = 0
            for df in in_dfs:
                if x_data_id == "reset_count":
                    df[x_data_id] = df[x_data_id]*parallelsims + i
                    i +=1
                else:
                    df[x_data_id] = df[x_data_id]*parallelsims
            parallelsims = 1

        for df in in_dfs:
            df["mean"] = df[y_data_id].rolling(avglen, center=True).mean()
            df["mean_cummax"] = df["mean"].cummax()

        def get_mean_cummax_mean(row):
            # print("row="+str(row))
            last_cummaxes = []
            for df in in_dfs:
                lim = row[x_data_id]
                # print("lim="+str(lim))
                df_uptonow = df.loc[df[x_data_id] < lim]
                # print("df_uptonow="+str(df_uptonow))
                if len(df_uptonow) > 0:
                    latest = df_uptonow[x_data_id].idxmax
                    last_cummax = df_uptonow.iloc[latest]["mean_cummax"]
                    last_cummaxes.append(last_cummax)
                else:
                    last_cummaxes.append(0)
            mean_cummax_mean = sum(last_cummaxes)/len(last_cummaxes)
            mean_cummax_std = math.sqrt(sum([lc*lc for lc in last_cummaxes])/len(last_cummaxes) - mean_cummax_mean*mean_cummax_mean)
            return mean_cummax_mean, mean_cummax_std

        if cummax:
            # This is so inefficient
            print("Computing mean cumulative mean max...")
            for df in in_dfs:
                df[["mean_cummax_mean","mean_cummax_std"]] = df.apply(get_mean_cummax_mean, axis = 1, result_type="expand")
            print("Done.")
            
        concatdf = pd.concat(in_dfs)
        concatdf = concatdf.sort_values(x_data_id)
        concatdf.reset_index(drop=True)
        rdf = pd.DataFrame()
        rdf[y_data_id] = concatdf[y_data_id]
        # print(f"df = \n{df}")
        # print(f"concatdf = \n{concatdf}")
        # print(f"parallel_runs*avglen = {parallel_runs}*{avglen} = {parallel_runs*avglen}")
        # print(f"rdf = \n{rdf}")
        rdf[x_data_id] = concatdf[x_data_id]
        if cummax:
            rdf["mean"] = concatdf["mean_cummax_mean"]
            if not nostd:
                rdf["std"] = concatdf["mean_cummax_std"]
            print("cumulative maximum mean performance: "+str(rdf["mean"].max()))
        else:
            rdf["mean"] = concatdf[y_data_id].rolling(parallel_runs*avglen, center=True).mean()
            if not nostd:
                rdf["std"] = concatdf[y_data_id].rolling(parallel_runs*avglen, center=True).std()
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     print(rdf)
            # max_loc = rdf["mean"].idxmax()
            print("maximum mean performance: "+str(rdf['mean'].max()))

        out_dfs.append(rdf)
    return out_dfs


def makePlot(dfs : List[pd.DataFrame],
             x_data_id : str,
             max_x : float,
             min_x : float,
             y_data_id : str,
             max_y : float,
             min_y : float,
             doAvg : bool = False,
             title : str = None,
             xlabel : str = None,
             ylabel : str = None,
             raw : bool = False,
             dfLabels : List[str] = None,
             cummax : bool = False):

    plt.clf()

    print(f"Plotting {len(dfs)} dfs")
    for df in dfs:
        df.reset_index(drop = True, inplace=True)
        # print(df.head())
    showLegend = True
    if dfLabels is None:
        dfLabels = [None]*len(dfs)
        showLegend = False


    sns.set_theme(style="ticks") #"darkgrid")
    sns.set_context("paper")

    # palette = sns.color_palette("tab10")#"husl", len(dfs))
    palette = sns.color_palette("husl", len(dfs))
    i = 0
    if raw:
        for df in dfs:
            print("cols = "+str(df.columns))
            c = palette[i]
            if doAvg:
                c = [(e+1)/2 for e in c]
            p = sns.lineplot(data=df,x=x_data_id,y=y_data_id,color=c, alpha = 0.7, ci=None) #, ax = ax) #
            i+=1
    i = 0
    if doAvg:                                
        for df in dfs:
            c = palette[i]
            print(f"Plotting {dfLabels[i]} mean")
            p = sns.lineplot(x=df[x_data_id],y=df["mean"], color=c, label=dfLabels[i], ci=None) #, ax = ax) #
            if not raw and "std" in df:
                print(f"Plotting {dfLabels[i]} std")
                cis = (df["mean"] - df["std"], df["mean"] + df["std"])
                c = [(e+1)/2 for e in c]
                p.fill_between(df[x_data_id],cis[0],cis[1], color=c, alpha = 0.5)
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
        p.legend(loc='upper left')
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

ap = argparse.ArgumentParser()
ap.add_argument("--csvfiles", nargs="+", required=True, type=str, help="Csv file(s) to read from")
ap.add_argument("--nogui", default=True, action='store_true', help="Dont show the plot window, just save to file")
ap.add_argument("--once", default=True, action='store_true', help="Plot only once")
ap.add_argument("--noavg", default=False, action='store_true', help="Do not plot curve average")
ap.add_argument("--avgfiles", default=False, action='store_true', help="Make an average pof the provided files instead of displaying all of them")
ap.add_argument("--raw", default=False, action='store_true', help="Plot raw data")
ap.add_argument("--maxx", required=False, default=None, type=float, help="Maximum x value to plot")
ap.add_argument("--minx", required=False, default=None, type=float, help="Minimum x value to plot")
ap.add_argument("--maxy", required=False, default=None, type=float, help="Maximum y axis value")
ap.add_argument("--miny", required=False, default=None, type=float, help="Minimum y axis value")
ap.add_argument("--period", required=False, default=5, type=float, help="Seconds to wait between plot update")
ap.add_argument("--out", required=False, default=None, type=str, help="Filename for the output plot")
ap.add_argument("--ydataid", required=False, default=None, type=str, help="Data to put on the y axis")
ap.add_argument("--xdataid", required=False, default="reset_count", type=str, help="Data to put on the x axis")
ap.add_argument("--avglen", required=False, default=20, type=int, help="Window size of running average")
ap.add_argument("--xscaling", required=False, default=1, type=float, help="Scale the x values by this factor")
ap.add_argument("--xlabel", required=False, default=None, type=str, help="label to put on x axis")
ap.add_argument("--ylabel", required=False, default=None, type=str, help="label to put on y axis")
ap.add_argument("--title", required=False, default=None, type=str, help="plot title")
ap.add_argument("--format", required=False, default="pdf", type=str, help="format of the output file")
ap.add_argument("--savedfs", default=False, action='store_true', help="Save prepped dataframes as csv")
ap.add_argument("--dontplot", default=False, action='store_true', help="Do not plot")
ap.add_argument("--loadprepped", default=False, action='store_true', help="load already prepared csv files")
ap.add_argument("--deparallelize", default=False, action='store_true', help="Transform data collected in parallel in sequential data")
ap.add_argument("--parallelsims", required=False, default=1, type=int, help="Number of parallel simulators per run")
ap.add_argument("--legend", nargs="+", required=False, default=None, type=str, help="List of the labels to put in the legend")
ap.add_argument("--cummax", default=False, action='store_true', help="Plot cumulative maximum")
ap.add_argument("--nostd", default=False, action='store_true', help="Dont plot std")
ap.add_argument("--outfname", required=False, default=None, type=str, help="Name of the output file (without path)")


ap.set_defaults(feature=True)
args = vars(ap.parse_args())
if not args["nogui"]:
    signal.signal(signal.SIGINT, signal_handler)

matplotlib.rcParams['figure.raise_window'] = False
#matplotlib.use('Tkagg')
if not args["nogui"]:
    plt.ion()
    plt.show()

if args["ydataid"] is not None:
    y_data_id = args["ydataid"]
else:
    y_data_id="ep_reward"


#fig, ax = plt.subplots(figsize=(11, 8.5))
while not ctrl_c_received:
    #print("Plotting")
    try:
        csvfiles = args["csvfiles"]
        commonPath = os.path.commonpath([os.path.abspath(os.path.dirname(cf)) for cf in csvfiles])
        commonRealPath = os.path.realpath(commonPath) # absolute path without links
        title = args["title"]
        if title is None:
            title = commonRealPath.split("/")[-2]+"/"+commonRealPath.split("/")[-1]
        if title.lower() == "none":
            title = None
        if not args["loadprepped"]:
            dfs = prepData(csvfiles=csvfiles,
                            x_data_id=args["xdataid"],
                            avgFiles=args["avgfiles"],
                            xscaling=args["xscaling"],
                            max_x = args["maxx"],
                            deparallelize = args["deparallelize"],
                            avglen=args["avglen"],
                            cummax = args["cummax"],
                            parallelsims= args["parallelsims"],
                            nostd=args["nostd"])
        else:
            dfs = [pd.read_csv(csvfile) for csvfile in csvfiles]
        if args["savedfs"]:
            i = 0
            for df in dfs:
                if args["outfname"] is None:
                    fname = y_data_id
                else:
                    fname = args["outfname"]
                if args["avgfiles"]:
                    fname+="_avg"
                path = commonPath+"/preppedDf_"+fname+"_"+str(i)+".csv"
                df.to_csv(path)
                print("Saved to "+path)
                i+=1
        if not args["dontplot"]:
            if args["legend"] is not None:
                dfLabels = args["legend"]
            elif args["avgfiles"]:
                dfLabels = None
            else:
                dfLabels = [None]*len(dfs)
                i = 0
                for csvfile in csvfiles:
                    for f in csvfile.split("/"):
                        if f.startswith("seed_"):
                            dfLabels[i] = f[5:]
                    if dfLabels[i] is None:
                        dfLabels[i] = chr(65+i)
                    i+=1

            makePlot(dfs,
                    x_data_id=args["xdataid"],
                    max_x = args["maxx"],
                    min_x = args["minx"],
                    y_data_id=y_data_id,
                    max_y = args["maxy"],
                    min_y= args["miny"],
                    doAvg = not args["noavg"],
                    title = title,
                    xlabel = args["xlabel"],
                    ylabel = args["ylabel"],
                    raw = args["raw"],
                    dfLabels=dfLabels,
                    cummax=args["cummax"])
            if args["out"] is not None:
                path = args["out"]
            else:
                if args["outfname"] is None:
                    fname = y_data_id
                else:
                    fname = args["outfname"]
                if args["avgfiles"]:
                    fname+="_avg"
                path = commonPath+"/"+fname+"."+args["format"]
            if args["format"] == "png":
                plt.savefig(path, dpi=600,bbox_inches='tight')
            else:
                plt.savefig(path,bbox_inches='tight')
            print("Saved to "+path)

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
    if args["once"]:
        break
    plt.pause(args["period"])
