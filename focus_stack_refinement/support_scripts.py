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
from git.focus_stack_refinement.focus_filtering.filter_image import measure_focus
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


def extract_specific_frames(path_in, path_out, frame_indices=None):
    """
    Creates directory at path_out. Receives video path file as path_in, generates a collection of specific frames
    and puts it to path_out.

    :param path_in: path to the video file
    :param path_out: path to the output directory
    :param frame_indices: array of seeked frame indices
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
    for i in frame_indices:
        vidcap.set(cv.CAP_PROP_POS_FRAMES, i)    # added this line
        cv.imwrite(path_out + "frame%d.jpg" % i, image)
        success, image = vidcap.read()
        if not success:
            return


def show_specific_frame(path_in, frame_index):
    """
        Receives video path file as path_in, shows the chosen frame in a separate window.
        Purpose: inspect whether the frame is focused.

        :param path_in: path to the video file
        :param frame_index: index of the frame to show
        :return:
        """
    vidcap = cv.VideoCapture(path_in)
    if vidcap is None:
        raise FileNotFoundError

    vidcap.set(cv.CAP_PROP_POS_FRAMES, frame_index)  # added this line
    success, image = vidcap.read()
    if not success:
        return
    cv.imshow(path_in, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


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


def get_classic_filtering_results(path_to_stack=None, path_to_video=None, measurement_stack=None, parts=True):
    stack = []
    if path_to_video is not None:
        stack = get_frames_as_array(path_to_video)
        # res = get_focus_results_from_array(stack, measure='TENG', to_grayscale=True)
    else:
        for frame_index in range(0, len(os.listdir(path_to_stack))):
            frame_name = f'frame{frame_index}.jpg'
            stack.append(cv.imread(path_to_stack + frame_name, cv.IMREAD_COLOR))
    return filter_focused_images(stack, 'TENG', parts=parts, count=8, measurement_stack=measurement_stack)


def preprocess_parts(z_stack, segment_count=8, clean_dust=True, measure='TENG'):
    preprocessing = {}
    preprocessing['first_frame'] = z_stack[0]
    (h, w, _) = z_stack[0].shape

    xseg_size = w // segment_count
    yseg_size = xseg_size

    best_frames = np.zeros((segment_count, segment_count), dtype=int)
    func = of.default_parts

    if clean_dust:
        dusted = z_stack[0].copy()

    mask = measure_focus.get_sample_seg_mask(z_stack[0])
    preprocessing['mask'] = mask

    func_res = func(z_stack[0], xseg_size, yseg_size, measure)

    avg_teng = []
    preprocessing['frame_measures'] = []
    for i, img in enumerate(z_stack):
        # find maximum values and images where sectors with such values are taken
        tmp_func_res = func(z_stack[i], xseg_size, yseg_size, measure)
        preprocessing['frame_measures'].append(tmp_func_res)
    return preprocessing


def get_focused_images_parts_using_preprocessed(
        preprocessed_object=None,
        measure='TENG',
        xseg_size=-1,
        yseg_size=-1,
        save_sectors=False,
        path_avg_teng_plot=None
):
    h = preprocessed_object['first_frame'].shape
    w = preprocessed_object['first_frame'].shape
    if xseg_size == -1 and yseg_size == -1:
        xseg_size = w // 8
    if xseg_size == -1 and yseg_size != -1:
        xseg_size = yseg_size
    if yseg_size == -1 and xseg_size != -1:
        yseg_size = xseg_size
    xseg_amount = w // xseg_size + (1 if w % xseg_size != 0 else 0)
    yseg_amount = h // yseg_size + (1 if h % yseg_size != 0 else 0)
    best_frames = np.zeros((yseg_amount, xseg_amount), dtype=int)
    func = measure_focus.default_parts

    mask = preprocessed_object['mask']

    func_res = func(preprocessed_object['first_frame'], xseg_size, yseg_size, measure)
    avg_teng = []

    for i, frame_measure in enumerate(preprocessed_object['frame_mesasures']):
        # find maximum values and images where sectors with such values are taken
        tmp_func_res = frame_measure

        avg_teng.append(np.average(tmp_func_res))

        measure_focus.update_best_segs(h, w, xseg_size, yseg_size, i, tmp_func_res, func_res, best_frames)

    # save best images and links to sectors which are best in them
    res = measure_focus.save_best_parts(h, w, xseg_size, yseg_size, mask, best_frames)

    if path_avg_teng_plot is not None:
        plt.plot(np.squeeze(avg_teng))
        name = (path_avg_teng_plot.split(".")[0])
        plot_name = "{}_avg_{}.png".format(name, measure)
        plt.savefig(plot_name)
        plt.close()

    res = list(res.keys())
    res.sort()
    return res, avg_teng


def rescale_frame(frame, scale_percentage=100):
    width = int(frame.shape[1] * scale_percentage / 100)
    height = int(frame.shape[0] * scale_percentage / 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA), width, height


# TODO: documentation; implement downscale; implement filter_window, relative to the stack size
def find_peak(output_plot_name=None, path_to_video=None, path_to_stack=None, z_stack=None,
              z_stack_measures=None, focus_measure='TENG', filter_window=11):
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
    elif z_stack_measures is not None:
        res = z_stack_measures
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

    return peak_lower_x, peak_lower_y


def find_peak_v3(output_plot_name=None, path_to_video=None, path_to_stack=None, z_stack=None,
              z_stack_measures=None, focus_measure='TENG', filter_window=11):

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
    elif z_stack_measures is not None:
        res = z_stack_measures
    else:
        raise Exception()

    res_smooth = sig.savgol_filter(res, filter_window, 2)
    res_half_size = len(res_smooth) // 2
    res_smooth_avg = np.min(res_smooth)   # TODO: чем дополнять? Кажется, нужно отражать, чтобы не было резких перепадов
    res_smooth = np.append(np.full(res_half_size, res_smooth_avg),
                           np.append(np.array(res_smooth),
                                     np.full(res_half_size, res_smooth_avg)))

    res_smooth_glob_min = np.min(res_smooth)
    res_smooth_glob_max = np.max(res_smooth)
    res_smooth_max_prominence = res_smooth_glob_max - res_smooth_glob_min

    start_prominence = 0
    end_prominence = res_smooth_max_prominence
    found_peaks_x, peak_props = sig.find_peaks(res_smooth, width=(0, None), prominence=(start_prominence, None))
    num_of_peaks = len(found_peaks_x)
    while num_of_peaks != 1 and start_prominence != end_prominence:
        mid_prominence = (end_prominence + start_prominence) // 2
        found_peaks_x, peak_props = sig.find_peaks(res_smooth, width=(0, None), prominence=(mid_prominence, None))
        num_of_peaks = len(found_peaks_x)
        if num_of_peaks >= 1:
            if start_prominence == mid_prominence:
                break
            start_prominence = mid_prominence
        else:
            end_prominence = mid_prominence

    peak_lower_x = int(found_peaks_x[0] - peak_props['widths'][0]) - res_half_size
    peak_lower_y = int(found_peaks_x[0] + peak_props['widths'][0]) + 1 - res_half_size
    peak_range = range(max(0, peak_lower_x), min(len(res), peak_lower_y))

    plt.plot(res)
    plt.plot(peak_range, np.array(res)[peak_range])
    if output_plot_name is not None:
        plt.savefig(f'results/peak_search/{output_plot_name}.png')
    else:
        plt.show()
    plt.close()

    return peak_lower_x, peak_lower_y

# incorrect: honeybee leg 320x240 medium (h), housefly compound eye fast (h), housefly compound eye medium (h)
#          Hydra fast (s), pollen gem fast, mutation of drosophila - fast (h/s), пыльник лилии fast (h/s), hd дафния (s)
#          почти треть из pro
# немного: butterfly wings ocales,
# ещё кажется, слишком узкий отрезок порой берёт


def find_peak_v4_mirror(output_plot_name=None, path_to_video=None, path_to_stack=None, z_stack=None,
                 z_stack_measures=None, focus_measure='TENG', filter_window=11, does_plotting=True):
    """
    Find peak and its width using signal.find_peak prominence binary search and edge-mirroring.

    :param output_plot_name: if provided, saves plot as {output_plot_name}.png
    :param path_to_video: 1st way to pass z-stack - video
    :param path_to_stack: 2nd way to pass z-stack - directory with pictures
    :param z_stack: 3d way to pass z-stack - list
    :param z_stack_measures: 4th way to pass z-stack - processed focus measure stack (list)
    :param focus_measure: focus measure to use
    :param filter_window: width of the filter window used by savgol-filter smoothing. Determines aggressiveness of the smoothing
    :param does_plotting: if set to false, no plot is shown nor saved
    :return: minimum and maximum index of the pruned segment
    """
    if path_to_stack is not None:
        stack = []
        for frame_id in range(0, len(os.listdir(path_to_stack))):
            stack_frame = cv.imread(path_to_stack + f'frame{frame_id}.jpg', cv.IMREAD_COLOR)
            stack_frame = cv.cvtColor(stack_frame, cv.COLOR_BGR2GRAY)
            stack_frame, w, h = rescale_frame(stack_frame, 100)
            stack.append(stack_frame)
        res = get_focus_results_from_array(stack, measure=focus_measure, to_grayscale=False)  # TODO: True
    elif path_to_video is not None:
        stack = get_frames_as_array(path_to_video)
        res = get_focus_results_from_array(stack, measure=focus_measure, to_grayscale=True)
    elif z_stack is not None:
        stack = z_stack
        res = get_focus_results_from_array(z_stack, measure=focus_measure, to_grayscale=True)
    elif z_stack_measures is not None:
        res = z_stack_measures
    else:
        raise Exception()

    res_smooth = sig.savgol_filter(res, filter_window, 2)
    res_half_size = len(res_smooth) // 2
    res_smooth_avg = np.min(res_smooth)  # TODO: чем дополнять? Кажется, нужно отражать, чтобы не было резких перепадов
    res_smooth = np.append(np.flip(res_smooth[0: res_half_size]),
                             np.append(np.array(res_smooth),
                                       np.flip(res_smooth[res_half_size:])))

    res_smooth_glob_min = np.min(res_smooth)
    res_smooth_glob_max = np.max(res_smooth)
    res_smooth_max_prominence = res_smooth_glob_max - res_smooth_glob_min

    start_prominence = 0
    end_prominence = res_smooth_max_prominence
    found_peaks_x, peak_props = sig.find_peaks(res_smooth, width=(0, None), prominence=(start_prominence, None))
    num_of_peaks = len(found_peaks_x)
    while num_of_peaks != 1 and start_prominence != end_prominence:
        mid_prominence = (end_prominence + start_prominence) // 2
        found_peaks_x, peak_props = sig.find_peaks(res_smooth, width=(0, None), prominence=(mid_prominence, None))
        num_of_peaks = len(found_peaks_x)
        if num_of_peaks >= 1:
            if start_prominence == mid_prominence:
                break
            start_prominence = mid_prominence
        else:
            end_prominence = mid_prominence

    def transform_back(x, y, thresh_1, thresh_2):
        if (x + y) / 2 < thresh_1:
            x_m = y + 2 * (thresh_1 - y)
            y_m = x + 2 * (thresh_1 - x)
        elif (x + y) / 2 >= thresh_2:
            x_m = y + 2 * (thresh_2 - y)
            y_m = x + 2 * (thresh_2 - x)
        else:
            x_m = x
            y_m = y
        return max(x_m, thresh_1) - thresh_1, min(y_m, thresh_2) - thresh_1

    peak_lower_x = int(found_peaks_x[0] - peak_props['widths'][0])
    peak_lower_y = int(found_peaks_x[0] + peak_props['widths'][0]) + 1

    peak_lower_x, peak_lower_y = transform_back(peak_lower_x, peak_lower_y, res_half_size, len(res) + res_half_size)

    peak_range = range(max(0, peak_lower_x), min(len(res), peak_lower_y))

    if does_plotting:
        plt.plot(res)
        plt.plot(peak_range, np.array(res)[peak_range])
        if output_plot_name is not None:
            plt.savefig(f'results/peak_search/{output_plot_name}.png')
        else:
            plt.show()
        plt.close()

    return peak_lower_x, peak_lower_y