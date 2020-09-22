import os
import cv2 as cv
from experiments.focus_map_getters import get_image_focus_map
import experiments.objective_functions as of

datasets_path = '../datasets/'
results_path = '../results/'
stack_name = 'Human Hair - HD'

file_names = os.listdir(datasets_path + stack_name)
window_size = 21
fms = {'LAPM', 'TENG', 'MLOG', 'VOLL4'}
is_masked = True

for fm in fms:
    for i in range(0, len(file_names)):
        img = cv.imread(datasets_path + stack_name + '/' + file_names[i], cv.IMREAD_COLOR)
        focus_map = get_image_focus_map(img, focus_operator_window_size=window_size,
                                             focus_operator=of.name_to_function(fm), is_focus_window_masked=is_masked)

        results_directory = f'{results_path}{stack_name} - FM.{fm}.{window_size}.{is_masked}'
        try:
            os.mkdir(results_directory)
        except FileExistsError:
            pass
        cv.imwrite(results_directory + '/' + file_names[i], focus_map)
