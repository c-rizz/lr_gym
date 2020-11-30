#!/usr/bin/env python3

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse
import time
import signal
import os

def makePlot(csvfile : str, x_data_id : str, max_x : float):
    df = pd.read_csv(csvfile)

    if max_x is not None:
        df = df.loc[df['time_from_start'] < max_x]

    avg_reward = df["ep_reward"].rolling(20).mean()


    sns.set_theme(style="darkgrid")
    #sns.set_style("dark")
    #sns.set_context("paper")
    sns.lineplot(data=df,x=x_data_id,y='ep_reward',color="#6699ff") #, ax = ax) #
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
ap.add_argument("--usetime", default=False, action='store_true', help="Use time on the x axis")
ap.add_argument("--useenvsteps", default=False, action='store_true', help="Use environment steps on the x axis")
ap.add_argument("--maxx", required=False, default=None, type=float, help="Maximum x value to plot")
ap.set_defaults(feature=True)
args = vars(ap.parse_args())
signal.signal(signal.SIGINT, signal_handler)

matplotlib.rcParams['figure.raise_window'] = False
#matplotlib.use('Tkagg')
if not args["nogui"]:
    plt.ion()
    plt.show()

if args["usetime"] and args["useenvsteps"]:
    print("You cannot use --usetime and --useenvsteps at the same time")
    exit(0)

if args["usetime"]:
    x_data_id = 'time_from_start'
elif args["useenvsteps"]:
    x_data_id = "total_steps"
else:
    x_data_id = 'reset_count'

#fig, ax = plt.subplots(figsize=(11, 8.5))
while not ctrl_c_received:
    makePlot(args["csvfile"], x_data_id, max_x = args["maxx"])
    plt.savefig(os.path.dirname(args["csvfile"])+"/reward.pdf")
    #plt.show(block=True)
    if not args["nogui"]:
        plt.draw()
        plt.pause(0.001)
    if args["once"]:
        plt.show(block=True)
        break
    time.sleep(1)
