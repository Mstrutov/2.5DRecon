import numpy as np
import cv2 as cv
import time
import os
import re
import pandas as pd
import git.experiments.objective_functions as of
import scipy.signal as sig
from matplotlib import pyplot as plt
from git.focus_stack_refinement.focus_filtering.filter_image import filter_focused_images
from shutil import rmtree


def extract_frames(path_in, path_out, distance_between_frames=1):
    """
    Receives video path file as path_in, generates a collection of frames to path_out

    :param path_in: path to the video file
    :param path_out: path to the output directory
    :param distance_between_frames: distance between consequent frames
    :return:
    """
    try:
        os.mkdir(path_out)
    except FileExistsError:
        pass
    vidcap = cv.VideoCapture(path_in)
    if vidcap is None:
        raise FileNotFoundError
    success, image = vidcap.read()
    if not success:
        return
    for i in range(0, int(vidcap.get(cv.CAP_PROP_FRAME_COUNT) - 1), distance_between_frames):
        vidcap.set(cv.CAP_PROP_POS_FRAMES, i)    # added this line
        cv.imwrite(path_out + "frame%d.jpg" % i, image)
        success, image = vidcap.read()
        if not success:
            return


def get_frames_as_array(path_in, distance_between_frames=1):
    """
        Receives video path file as path_in, outputs an array of frames

        :param path_in: path to the video file
        :param distance_between_frames: distance between consequent frames
        :return: array of frames
    """
    vidcap = cv.VideoCapture(path_in)
    images = []
    success, image = vidcap.read()
    if not success:
        return images
    images.append(image)
    for i in range(0, int(vidcap.get(cv.CAP_PROP_FRAME_COUNT) - 1), distance_between_frames):
        vidcap.set(cv.CAP_PROP_POS_FRAMES, i)    # added this line
        success, image = vidcap.read()
        if not success:
            return images
        images.append(image)
    return images


def get_focus_results(path_to_stack, measure='TENG', to_grayscale=False):
    """
    Applies focus measure operator to the stack, outputs one-dimensional array consisting of frames' focus measures

    :param path_to_stack: path to the stack directory
    :param measure: focus measure operator name. Should be provided on objective_functions.name_to_function()
    :param to_grayscale: whether to convert frames to grayscale
    :return: one-dimensional array consisting of frames' focus measures
    """
    func_res = []
    func = of.name_to_function(measure)

    for frame_index in range(0, len(os.listdir(path_to_stack))):
        frame_name = f'frame{frame_index}.jpg'
        frame = cv.imread(path_to_stack + frame_name, cv.IMREAD_COLOR)

        if to_grayscale:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        func_res.append(func(frame))

    return func_res


def get_focus_results_from_array(stack_array, measure='TENG', to_grayscale=False):
    """
    Applies focus measure operator to the stack, outputs one-dimensional array consisting of frames' focus measures

    :param stack_array: z-stack
    :param measure: focus measure operator name. Should be provided on objective_functions.name_to_function()
    :param to_grayscale: whether to convert frames to grayscale
    :return: one-dimensional array consisting of frames' focus measures
    """
    func_res = []
    func = of.name_to_function(measure)

    for frame in stack_array:
        if to_grayscale:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        func_res.append(func(frame))

    return func_res


def draw_plt_for_measures(path_to_dataset, dataset_name, graph_name, focus_measures, generate_tmp_frame_stack=False):
    """
    Receives a path to the video, gets focus measure results for each focus measure operator, draws and saves plots.
    Poorly optimized version.

    :param path_to_dataset: a path to a directory with the video
    :param dataset_name: name of the dataset video
    :param graph_name: final plot name
    :param focus_measures: array of focus measure names from objective_functions
    :param generate_tmp_frame_stack: currently does nothing
    :return:
    """
    ax = None
    plt.figure(num=None, figsize=(6, 10))
    time_res = []
    frames = get_frames_as_array(path_to_dataset + dataset_name)
    print('got ' + str(len(frames)) + ' frames')
    for focus_measure_i in range(0, len(focus_measures)):
        start_time = time.time()
        if ax is not None:
            ax = plt.subplot(len(focus_measures), 1, focus_measure_i+1, sharex=ax)
        else:
            ax = plt.subplot(len(focus_measures), 1, focus_measure_i+1)
            ax.set_title(dataset_name)

        ax.set_ylabel(focus_measures[focus_measure_i])

        # if generate_tmp_frame_stack:

        # res = get_focus_results(path_to_dataset + 'tmp/', measure=focus_measures[focus_measure_i], to_grayscale=False)
        res = get_focus_results_from_array(frames, focus_measures[focus_measure_i], to_grayscale=False)
        print('got res:' + str(res))
        # if generate_tmp_frame_stack:
        #    rmtree(path_to_dataset + 'tmp/')

        plt.plot(res)
        if focus_measure_i != len(focus_measures) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        time_res.append(time.time() - start_time)

    plt.tight_layout()
    plt.savefig(f'results/{graph_name}.png')
    #plt.show()
    plt.close()

    return time_res


def draw_plt_optimized(path_to_dataset, dataset_name, graph_name, focus_measures, generate_tmp_frame_stack=False):
    """
    Receives a path to the video, gets focus measure results for each focus measure operator, draws and saves plots.
    An OK-optimized version.

    :param path_to_dataset: a path to a directory with the video
    :param dataset_name: name of the dataset video
    :param graph_name: final plot name
    :param focus_measures: array of focus measure names from objective_functions
    :param generate_tmp_frame_stack: currently does nothing
    :return:
    """
    ax = None
    plt.figure(num=None, figsize=(6, 10))
    time_res = []
    focus_measures_res = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    vidcap = cv.VideoCapture(path_to_dataset + dataset_name)
    images = []

    for i in range(0, int(vidcap.get(cv.CAP_PROP_FRAME_COUNT) - 1), 1):
        vidcap.set(cv.CAP_PROP_POS_FRAMES, i)  # added this line
        success, image = vidcap.read()
        if not success:
            break

        for focus_measure_i in range(0, len(focus_measures)):
            if True:
                cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            func = of.name_to_function(focus_measures[focus_measure_i])
            focus_measures_res[focus_measure_i].append(func(image))
    #print(focus_measures_res)
    for focus_measure_i in range(0, len(focus_measures)):
        if ax is not None:
            ax = plt.subplot(len(focus_measures), 1, focus_measure_i + 1, sharex=ax)
        else:
            ax = plt.subplot(len(focus_measures), 1, focus_measure_i + 1)
            ax.set_title(dataset_name)

        ax.set_ylabel(focus_measures[focus_measure_i])

        plt.plot(focus_measures_res[focus_measure_i])
        if focus_measure_i != len(focus_measures) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig(f'results/{graph_name}.png')
    # plt.show()
    plt.close()


def get_classic_filtering_results(path_to_stack):
    stack = []
    for frame_index in range(0, len(os.listdir(path_to_stack))):
        frame_name = f'frame{frame_index}.jpg'
        stack.append(cv.imread(path_to_stack + frame_name, cv.IMREAD_COLOR))
    return filter_focused_images(stack, 'TENG', parts=True, count=5)


# TODO: documentation; implement downscale; implement filter_window, relative to the stack size
def find_peak(output_plot_name=None, path_to_video=None, path_to_stack=None, z_stack=None, focus_measure='TENG', filter_window=11):
    def rescale_frame(frame, scale_percentage=100):
        width = int(frame.shape[1] * scale_percentage / 100)
        height = int(frame.shape[0] * scale_percentage / 100)
        dim = (width, height)
        return cv.resize(frame, dim, interpolation=cv.INTER_AREA), width, height

    if path_to_stack is not None:
        stack = []
        for frame_id in range(0, len(os.listdir(path_to_stack))):
            stack_frame = cv.imread(path_to_stack + f'frame{frame_id}.jpg', cv.IMREAD_COLOR)
            stack_frame = cv.cvtColor(stack_frame, cv.COLOR_BGR2GRAY)
            stack_frame, w, h = rescale_frame(stack_frame, 100)
            stack.append(stack_frame)
        res = get_focus_results_from_array(stack, measure=focus_measure, to_grayscale=False)    # TODO: True
    elif path_to_video is not None:
        stack = get_frames_as_array(path_to_video)
        res = get_focus_results_from_array(stack, measure=focus_measure, to_grayscale=True)
    elif z_stack is not None:
        stack = z_stack
        res = get_focus_results_from_array(z_stack, measure=focus_measure, to_grayscale=True)
    else:
        raise Exception()

    res_smooth = sig.savgol_filter(res, min(filter_window, len(res)), 2)
    res_smooth[-1] = np.mean(res_smooth[-3:-2])
    res_smooth[0] = np.mean(res_smooth[1:2])
    found_peaks_x, peak_props = sig.find_peaks(res_smooth, distance=2*len(res), width=(1, None))
    if len(found_peaks_x) == 0:
        raise RuntimeError()
    peak_lower_x = max(0, int(found_peaks_x - peak_props['widths']))
    peak_lower_y = min(int(found_peaks_x + peak_props['widths']) + 1, len(res))
    peak_range = range(peak_lower_x, peak_lower_y)

    plt.plot(res)
    plt.plot(peak_range, np.array(res)[peak_range])
    if output_plot_name is not None:
        plt.savefig(f'results/peak_search/{output_plot_name}.png')
    else:
        plt.show()
    plt.close()
