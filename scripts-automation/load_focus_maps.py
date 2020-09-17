import os
import cv2 as cv
from experiments.focus_map_getters import get_image_focus_map
import experiments.objective_functions as of

datasets_path = '../datasets/'
results_path = '../results/'
stack_name = 'Human Hair'

for image_name in os.listdir(datasets_path + stack_name):
    img = cv.imread(datasets_path + stack_name + '/' + image_name, cv.IMREAD_COLOR)
    focus_map = get_image_focus_map(img, focus_operator=of.TENG)
    cv.imwrite(results_path + 'FM.TENG - ' + stack_name + '/' + image_name, focus_map)