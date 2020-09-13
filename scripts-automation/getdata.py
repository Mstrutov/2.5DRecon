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


# if __file_name__=="__main__":
#     a = argparse.ArgumentParser()
#     a.add_argument("--pathIn", help="path to video")
#     a.add_argument("--pathOut", help="path to images")
#     args = a.parse_args()
#     print(args)
#inputPath = r"datasets\\videos\\Drosophila\\super_slow.mp4"
#outputPath = r"datasets\\Drosophila\\"
#extractImages(inputPath, outputPath)

input_dir = r'datasets\\videos\\'
output_dir = r'datasets\\'

# getting the list of files
file_names = os.listdir(input_dir)
file_names = np.array(file_names)

# selecting '...<number>.mp4' files
mask = [re.match(r'.*\.mp4', file_name) is not None for file_name in file_names]
videos = file_names[mask]
videos = []

for video in videos:
    # getting the source video name
    video_name = re.sub(r'(.*)\.mp4', r'\1', video)
    # creating directory for the output
    os.mkdir(output_dir + video_name)

    # extracting images from video
    input_path = input_dir + video_name + '.mp4'
    output_path = output_dir + video_name + r'\\'
    extractImages(input_path, output_path, msecsBetweenFrames=500)

extractImages('datasets/videos/Bee wing (super slow).mp4', 'datasets/Bee wing new 100/', msecsBetweenFrames=100)