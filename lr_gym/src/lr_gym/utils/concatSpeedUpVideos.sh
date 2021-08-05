#!/bin/bash
#Useful script to put together video files produced by RecorderGymEnvWrapper
#For some reasons leads to wrong duration metadata (or something like that)

rm all10fps.mp4
printf "file '%s'\n" *.mp4 > allmp4Vids.txt
ffmpeg -f concat -safe 0 -i allmp4Vids.txt -c copy all.mp4
ffmpeg -i all.mp4 -filter:v "setpts=PTS/10,fps=10" all10fps.mp4
rm allmp4Vids.txt