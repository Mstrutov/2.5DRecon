import numpy as np
import experiments.objective_functions as of
import experiments.focus_map_getters as fmg


def next_contour(grayscale_texture, mask, raw_contour, contour_mass_centre, window_size=7, first_step=1):
    diffs = contour_mass_centre - raw_contour
    maxes = np.amax(np.abs(diffs), axis=1, keepdims=True)
    smallest_steps = np.divide(diffs, maxes)
    raw_contour = np.append(raw_contour, smallest_steps, axis=1)
    raw_contour = np.append(raw_contour, maxes, axis=1)

    comprehensive_mask = np.zeros((grayscale_texture.shape[0] + window_size,
                                   grayscale_texture.shape[1] + window_size), np.uint8)
    half_window_size = (window_size - 1) // 2
    comprehensive_mask[half_window_size:(grayscale_texture.shape[0] + half_window_size),
        half_window_size:(grayscale_texture.shape[1] + half_window_size)] = mask

    raw_next_contour = []
    for contour_point in raw_contour:
        inner_points = np.array(contour_point[0:2], ndmin=2).transpose() + \
                       np.array(contour_point[2:4], ndmin=2).transpose() * \
                       np.tile(np.array(range(first_step, contour_point[4].astype(int) + 1)), (2, 1))
        inner_points = inner_points.transpose()

        focus_measure_values = []
        for inner_point in inner_points:
            inner_point = inner_point.astype(int) #x, y

            focus_measure_values.append(fmg.get_focus_in_pixel(grayscale_texture, mask, comprehensive_mask,
                                                               inner_point[0], inner_point[1], window_size, of.LAPM,
                                                               is_focus_window_masked=True))
        (point_x, point_y) = inner_points[np.argmax(focus_measure_values)].astype(int)
        raw_next_contour.append([point_x, point_y])
    return raw_next_contour
