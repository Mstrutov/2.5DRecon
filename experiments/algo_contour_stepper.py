import numpy as np
import experiments.objective_functions as of


def next_contour(grayscale_texture, raw_contour, contour_mass_centre, window_size=7):
    diffs = contour_mass_centre - raw_contour
    maxes = np.amax(np.abs(diffs), axis=1, keepdims=True)
    smallest_steps = np.divide(diffs, maxes)
    raw_contour = np.append(raw_contour, smallest_steps, axis=1)
    raw_contour = np.append(raw_contour, maxes, axis=1)

    max_w = grayscale_texture.shape[1]
    max_h = grayscale_texture.shape[0]

    raw_next_contour = []
    for contour_point in raw_contour:
        inner_points = np.array(contour_point[0:2], ndmin=2).transpose() + \
                       np.array(contour_point[2:4], ndmin=2).transpose() * \
                       np.tile(np.array(range(1, contour_point[4].astype(int) + 1)), (2, 1))
        inner_points = inner_points.transpose()

        focus_measure_values = []
        for inner_point in inner_points:
            inner_point = inner_point.astype(int)
            lower_x = max(0, inner_point[0] - (window_size - 1) // 2)
            upper_x = min(max_w, inner_point[0] + (window_size - 1) // 2) + 1
            lower_y = max(0, inner_point[1] - (window_size - 1) // 2)
            upper_y = min(max_h, inner_point[1] + (window_size - 1) // 2) + 1

            focus_measure_values.append(of.LAPM(grayscale_texture[lower_y:upper_y, lower_x:upper_x]))
        (point_x, point_y) = inner_points[np.argmax(focus_measure_values)].astype(int)
        raw_next_contour.append([point_x, point_y])
    return raw_next_contour
