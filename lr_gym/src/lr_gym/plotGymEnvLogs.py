#!/usr/bin/env python3

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse
import time
import signal
import os
import pandas
import numpy as np
from typing import List

def makePlot(csvfiles : List[str],
             x_data_id : str,
             max_x : float,
             min_x : float,
             y_data_id : str,
             max_y : float,
             min_y : float,
             doAvg : bool = False,
             title : str = "",
             avglen : int  = 20,
             noRaw : bool = False):
    plt.clf()

    multiaxes = x_data_id is None
    if multiaxes:
        x_data_id = "reset_count"

    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("husl", len(csvfiles))
    i = 0
    if not noRaw:
        for csvfile in csvfiles:
            df = pd.read_csv(csvfile)
            if "success" in df:
                df["success"] = df["success"].astype(int) # boolean to 0-1
            if max_x is not None:
                df = df.loc[df[x_data_id] < max_x]
            c = palette[i]
            if doAvg:
                c = [(e+1)/2 for e in c]
            p = sns.lineplot(data=df,x=x_data_id,y=y_data_id,color=c, alpha = 0.7) #, ax = ax) #
            i+=1
    i = 0
    if doAvg:                
        for csvfile in csvfiles:
            df = pd.read_csv(csvfile)
            if "success" in df:
                df["success"] = df["success"].astype(int) # boolean to 0-1
            if max_x is not None:
                df = df.loc[df[x_data_id] < max_x]
            avg_reward = df[y_data_id].rolling(avglen, center=True).mean()
            c = palette[i]
            p = sns.lineplot(x=df[x_data_id],y=avg_reward, color=c) #, ax = ax) #
            i+=1
    #plt.legend(loc='lower right', labels=names)
    # pathSplitted = os.path.dirname(csvfile).split("/")
    # plt.title(pathSplitted[-2]+"/"+pathSplitted[-1]+"/"+os.path.basename(csvfile))

    p.set_xlim(min_x,max_x) # If None they leave the current limit

    data_miny = df[y_data_id].min()
    data_maxy = df[y_data_id].max()
    if max_y is not None or min_y is not None:
        if min_y is None:
            min_y = data_miny - 0.05 * (data_maxy - data_miny)
        if max_y is None:
            max_y = data_maxy
        p.set_ylim(min_y,max_y)

    if multiaxes:
        additional_xdataids = ["total_steps", "time_from_start"]
        x_label = x_data_id
        i = 1
        for adx in additional_xdataids:
            ax2 = p.twiny()
            x2 = df[adx]
            p2 = sns.lineplot(x=x2, y=np.arange(len(x2)), visible=False)
            # Move twinned axis ticks and label from top to bottom
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")
            ax2.spines['bottom'].set_position(('outward', 10 * i))
            ax2.spines['bottom'].set_visible(False)
            plt.tick_params(which='both', bottom=False)
            ax2.set_xlabel("")
            ax2.grid(False)
            i+=1
            x_label = x_label + ", "+adx

        p.set_xlabel(x_label, labelpad = 10 * len(additional_xdataids))

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
ap.add_argument("--noraw", default=False, action='store_true', help="Do not plot raw data")
ap.add_argument("--maxx", required=False, default=None, type=float, help="Maximum x value to plot")
ap.add_argument("--minx", required=False, default=None, type=float, help="Minimum x value to plot")
ap.add_argument("--maxy", required=False, default=None, type=float, help="Maximum y axis value")
ap.add_argument("--miny", required=False, default=None, type=float, help="Minimum y axis value")
ap.add_argument("--period", required=False, default=5, type=float, help="Seconds to wait between plot update")
ap.add_argument("--out", required=False, default=None, type=str, help="Filename for the output plot")
ap.add_argument("--ydataid", required=False, default=None, type=str, help="Data to put on the y axis")
ap.add_argument("--xdataid", required=False, default=None, type=str, help="Data to put on the x axis")
ap.add_argument("--avglen", required=False, default=20, type=int, help="Window size of running average")
ap.add_argument("--format", required=False, default="pdf", type=str, help="format of the output file")

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

x_data_id = args["xdataid"]

#fig, ax = plt.subplots(figsize=(11, 8.5))
while not ctrl_c_received:
    #print("Plotting")
    try:
        csvfiles = args["csvfiles"]
        commonPath = os.path.commonpath([os.path.dirname(cf) for cf in csvfiles])
        title = commonPath.split("/")[-1]
        makePlot(csvfiles,
                 x_data_id,
                 max_x = args["maxx"],
                 min_x = args["minx"],
                 y_data_id=y_data_id,
                 max_y = args["maxy"],
                 min_y= args["miny"],
                 doAvg = not args["noavg"],
                 title = title,
                 avglen=args["avglen"],
                 noRaw = args["noraw"])
        if args["out"] is not None:
            fname = args["out"]
            if fname.split(".")[-1] == "png":
                plt.savefig(fname, dpi=1200)
            else:
                plt.savefig(fname)
        else:
            if args["format"] == "png":
                plt.savefig(commonPath+"/"+y_data_id+"."+args["format"], dpi=1200)
            else:
                plt.savefig(commonPath+"/"+y_data_id+"."+args["format"])

        #plt.show(block=True)
        if not args["nogui"]:
            plt.draw()
            plt.pause(0.01)
            if args["once"]:
                plt.show(block=True)
                break
    except pandas.errors.EmptyDataError:
        print("No data...")
    except FileNotFoundError:
        print("File not present...")
    if args["once"]:
        break
    plt.pause(args["period"])
