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

def makePlot(csvfile : str, x_data_id : str, max_x : float, y_data_id : str):
    df = pd.read_csv(csvfile)

    if max_x is not None:
        df = df.loc[df[x_data_id] < max_x]

    avg_reward = df[y_data_id].rolling(20).mean()

    plt.clf()
    sns.set_theme(style="darkgrid")
    #sns.set_style("dark")
    #sns.set_context("paper")
    sns.lineplot(data=df,x=x_data_id,y=y_data_id,color="#6699ff") #, ax = ax) #
    p = sns.lineplot(x=df[x_data_id],y=avg_reward, color="#0066cc") #, ax = ax) #
    #plt.legend(loc='lower right', labels=names)
    plt.title(os.path.dirname(args["csvfile"]).split("/")[-1]+"/"+os.path.basename(csvfile))
    if max_x is not None:
        p.set_xlim(-10,max_x)
    plt.tight_layout()



ctrl_c_received = False
def signal_handler(sig, frame):
    #print('You pressed Ctrl+C!')
    global ctrl_c_received
    ctrl_c_received = True

ap = argparse.ArgumentParser()
ap.add_argument("--csvfile", required=True, type=str, help="Csv file to read from")
ap.add_argument("--nogui", default=False, action='store_true', help="Dont show the plot window, just save to file")
ap.add_argument("--once", default=False, action='store_true', help="Plot only once")
ap.add_argument("--usetimex", default=False, action='store_true', help="Use time on the x axis")
ap.add_argument("--useenvstepsx", default=False, action='store_true', help="Use environment steps on the x axis")
ap.add_argument("--maxx", required=False, default=None, type=float, help="Maximum x value to plot")
ap.add_argument("--period", required=False, default=5, type=float, help="Seconds to wait between plot update")
ap.set_defaults(feature=True)
args = vars(ap.parse_args())
signal.signal(signal.SIGINT, signal_handler)

matplotlib.rcParams['figure.raise_window'] = False
#matplotlib.use('Tkagg')
if not args["nogui"]:
    plt.ion()
    plt.show()

if args["usetimex"] and args["useenvstepsx"]:
    print("You cannot use --usetimex and --useenvstepsx at the same time")
    exit(0)

if args["usetimex"]:
    x_data_id = 'time_from_start'
elif args["useenvstepsx"]:
    x_data_id = "total_steps"
else:
    x_data_id = 'reset_count'

#fig, ax = plt.subplots(figsize=(11, 8.5))
while not ctrl_c_received:
    #print("Plotting")
    try:
        makePlot(args["csvfile"], x_data_id, max_x = args["maxx"], y_data_id="ep_reward")
        plt.savefig(os.path.dirname(args["csvfile"])+"/reward.pdf")
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
