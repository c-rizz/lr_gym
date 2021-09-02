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
import numpy as np
from typing import List

def prepData(csvfiles : List[str],
             x_data_id : str,
             avgFiles : bool = False,
             xscaling : float = None,
             max_x : float = None,
             deparallelize : bool = False,
             avglen : int  = 20):

    in_dfs = [pd.read_csv(csvfile) for csvfile in csvfiles]
    parallel_runs = len(in_dfs)
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
            rdf["mean"] = df[y_data_id].rolling(avglen).mean()
            rdf["std"] = df[y_data_id].rolling(avglen).std()
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
                    df[x_data_id] = df[x_data_id]*parallel_runs + i
                    i +=1
                else:
                    df[x_data_id] = df[x_data_id]*parallel_runs
            parallel_runs = 1

        concatdf = pd.concat(in_dfs)
        concatdf = concatdf.sort_values(x_data_id)
        rdf = pd.DataFrame()
        rdf[y_data_id] = concatdf[y_data_id]
        rdf["mean"] = concatdf[y_data_id].rolling(parallel_runs*avglen).mean()
        rdf["std"] = concatdf[y_data_id].rolling(parallel_runs*avglen).std()
        rdf[x_data_id] = concatdf[x_data_id]
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
             dfLabels : List[str] = None):

    plt.clf()

    showLegend = True
    if dfLabels is None:
        dfLabels = [None]*len(dfs)
        showLegend = False


    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("husl", len(dfs))
    i = 0
    if raw:
        for df in dfs:
            print("cols = "+str(df.columns))
            c = palette[i]
            if doAvg:
                c = [(e+1)/2 for e in c]
            p = sns.lineplot(data=df,x=x_data_id,y=y_data_id,color=c, alpha = 0.7) #, ax = ax) #
            i+=1
    i = 0
    if doAvg:                                
        for df in dfs:
            c = palette[i]
            p = sns.lineplot(data=df,x=x_data_id,y="mean", color=c, label=dfLabels[i]) #, ax = ax) #
            if not raw:
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
        p.legend()

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
ap.add_argument("--nogui", default=False, action='store_true', help="Dont show the plot window, just save to file")
ap.add_argument("--once", default=False, action='store_true', help="Plot only once")
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
ap.add_argument("--legend", nargs="+", required=False, default=None, type=str, help="List of the labels to put in the legend")


ap.set_defaults(feature=True)
args = vars(ap.parse_args())
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
        title = args["title"]
        if title is None:
            title = commonPath.split("/")[-1]
        if title.lower() == "none":
            title = None
        if not args["loadprepped"]:
            dfs = prepData(csvfiles=csvfiles,
                            x_data_id=args["xdataid"],
                            avgFiles=args["avgfiles"],
                            xscaling=args["xscaling"],
                            max_x = args["maxx"],
                            deparallelize = args["deparallelize"],
                            avglen=args["avglen"])
        else:
            dfs = [pd.read_csv(csvfile) for csvfile in csvfiles]
        if args["savedfs"]:
            i = 0
            for df in dfs:
                path = commonPath+"/preppedDf_"+str(i)+".csv"
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
                    dfLabels=dfLabels)
            if args["out"] is not None:
                fname = args["out"]
                if fname.split(".")[-1] == "png":
                    plt.savefig(fname, dpi=1200)
                else:
                    plt.savefig(fname)
            else:
                fname = y_data_id
                if args["avgfiles"]:
                    fname+="_avg"
                path = commonPath+"/"+fname+"."+args["format"]
                if args["format"] == "png":
                    plt.savefig(path, dpi=1200)
                else:
                    plt.savefig(path)
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
