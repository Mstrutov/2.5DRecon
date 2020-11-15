import os
import cv2 as cv
from experiments.focus_map_getters import get_image_focus_map
import experiments.objective_functions as of

datasets_path = '../../datasets/'
results_path = '../results/'
stack_name = 'Bee wing new by-hand'

file_names = os.listdir(datasets_path + stack_name)
window_size = 21
fms = ['LAPM', 'TENG', 'MLOG', 'VOLL4']
is_masked = True
# prev_focus_map = None

for fm in fms:
    prev_focus_map = None
    for i in range(0, len(file_names) - 1):
        img = cv.imread(datasets_path + stack_name + '/' + file_names[i], cv.IMREAD_COLOR)
        focus_map = get_image_focus_map(img, focus_operator_window_size=window_size,
                                             focus_operator=of.name_to_function(fm), is_focus_window_masked=is_masked)
        if prev_focus_map is not None:
            focus_map_diff = cv.absdiff(focus_map, prev_focus_map)
            results_directory = f'{results_path}{stack_name} - FMDIFF.{fm}.{window_size}.{is_masked}'
            try:
                os.mkdir(results_directory)
            except FileExistsError:
                pass
            cv.imwrite(results_directory + '/' + file_names[i], focus_map_diff)
        prev_focus_map = focus_map
