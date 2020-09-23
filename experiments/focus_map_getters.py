import cv2 as cv
import numpy as np

import experiments.objective_functions as of
from experiments.apply_binarization import get_mask


def get_focus_in_pixel(grayscale_img, mask, comprehensive_mask,
                       x, y, window_size, focus_operator, is_focus_window_masked=True):
    half_window_size = (window_size - 1) // 2

    lower_x = max(0, x - half_window_size)
    upper_x = min(grayscale_img.shape[1], x + half_window_size) + 1
    lower_y = max(0, y - half_window_size)
    upper_y = min(grayscale_img.shape[0], y + half_window_size) + 1

    window = np.zeros((window_size, window_size), np.uint8)
    window = window + np.mean(
        grayscale_img[lower_y:upper_y, lower_x:upper_x][mask[lower_y:upper_y, lower_x:upper_x] == 255])

    np.putmask(window, comprehensive_mask[y:(y + window_size), x:(x + window_size)] == 255,
               grayscale_img[lower_y:upper_y, lower_x:upper_x])
    if is_focus_window_masked:
        return focus_operator(window)
    else:
        return focus_operator(grayscale_img[lower_y:upper_y, lower_x:upper_x])


def get_image_focus_map(img, erosion_dilation_kernel_size=5, iterations_of_erosion_dilation=6,
                        global_black_background_threshold=2, focus_operator_window_size=11, focus_operator=of.LAPM,
                        is_focus_window_masked=True):
    grayscale_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    mask = get_mask(grayscale_img, global_black_background_threshold)
    kernel = np.ones((erosion_dilation_kernel_size, erosion_dilation_kernel_size), np.uint8)
    erosion = cv.erode(mask, kernel, iterations=iterations_of_erosion_dilation)
    dilation = cv.dilate(erosion, kernel, iterations=iterations_of_erosion_dilation)
    mask = dilation

    raw_mask = mask.reshape(-1, 2)

    focus_map = np.zeros((grayscale_img.shape[0], grayscale_img.shape[1]), np.uint8)
    window_size = focus_operator_window_size

    # comprehensive mask is needed to process image's close-to-bounds pixels
    comprehensive_mask = np.zeros((img.shape[0] + window_size, img.shape[1] + window_size), np.uint8)
    half_window_size = (window_size - 1) // 2
    comprehensive_mask[half_window_size:(img.shape[0] + half_window_size),
        half_window_size:(img.shape[1] + half_window_size)] = mask

    for (y, x) in np.array(np.where(mask == 255)).transpose().reshape(-1, 2):
        focus_map[y, x] = get_focus_in_pixel(grayscale_img, mask, comprehensive_mask,
                                             x, y, window_size, focus_operator, is_focus_window_masked)
    focus_map = focus_map / np.amax(focus_map) * 255
    return focus_map


def get_stack_focus_maps(img_array, erosion_dilation_kernel_size=5, iterations_of_erosion_dilation=6,
                         global_black_background_threshold=2, focus_operator_window_size=11, focus_operator=of.LAPM):
    resulting_focus_map_stack = []
    for i in range(0, len(img_array)):
        img = img_array[i]
        resulting_focus_map_stack.append(get_image_focus_map(
            img, erosion_dilation_kernel_size, iterations_of_erosion_dilation,
            global_black_background_threshold, focus_operator_window_size, focus_operator
        ))
    return resulting_focus_map_stack
