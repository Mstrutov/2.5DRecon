import argparse
import cv2
import os
import re
import numpy as np


def extractImages(pathIn, pathOut, msecsBetweenFrames=200):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*msecsBetweenFrames))    # added this line
        cv2.imwrite(pathOut + "frame%d.jpg" % count, image)
        success, image = vidcap.read()
        #cv2.imshow('Live', image)
        print('Read a new frame: ', success)
            # save frame as JPEG file
        count = count + 1


def extract_frames(path_in, path_out, distance_between_frames=1):
    vidcap = cv2.VideoCapture(path_in)
    success, image = vidcap.read()
    for i in range(0, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), distance_between_frames):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, i)    # added this line
        cv2.imwrite(path_out + "frame%d.jpg" % i, image)
        success, image = vidcap.read()


# if __file_name__=="__main__":
#     a = argparse.ArgumentParser()
#     a.add_argument("--pathIn", help="path to video")
#     a.add_argument("--pathOut", help="path to images")
#     args = a.parse_args()
#     print(args)
#inputPath = r"datasets\\videos\\Drosophila\\super_slow.mp4"
#outputPath = r"datasets\\Drosophila\\"
#extractImages(inputPath, outputPath)

input_dir = r'../../datasets/videos/new/'
output_dir = r'../../datasets/thick_specimen/'

# getting the list of files
file_names = os.listdir(input_dir)
file_names = np.array(file_names)

# selecting '...<number>.mp4' files
mask = [re.match(r'.*\.wmv', file_name) is not None for file_name in file_names]
videos = file_names[mask]

for video in videos:
    # getting the source video name
    video_name = re.sub(r'(.*)\.wmv', r'\1', video)
    # creating directory for the output
    os.mkdir(output_dir + video_name)

    # extracting images from video
    input_path = input_dir + video_name + '.wmv'
    output_path = output_dir + video_name + r'/'
    extract_frames(input_path, output_path, distance_between_frames=1)
    # extractImages(input_path, output_path, msecsBetweenFrames=33)

#extractImages('datasets/videos/Bee wing (super slow).mp4', 'datasets/Bee wing new 100/', msecsBetweenFrames=100)