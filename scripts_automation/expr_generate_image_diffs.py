import os
import cv2 as cv
import numpy as np
from experiments.focus_map_getters import get_image_focus_map
import experiments.objective_functions as of

datasets_path = '../../datasets/'
results_path = '../results/'
stack_name = 'Human hair by-hand'

file_names = os.listdir(datasets_path + stack_name)

prev_grayscale_img = None
for i in range(0, len(file_names) - 1):
    img = cv.imread(datasets_path + stack_name + '/' + file_names[i], cv.IMREAD_COLOR)
    grayscale_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    if prev_grayscale_img is not None:
        img_diff = cv.absdiff(grayscale_img, prev_grayscale_img)
        img_diff = img_diff / np.amax(img_diff) * 255
        results_directory = f'{results_path}{stack_name} - IMGDIFF'
        try:
            os.mkdir(results_directory)
        except FileExistsError:
            pass
        cv.imwrite(results_directory + '/' + file_names[i], img_diff)
    prev_grayscale_img = grayscale_img
