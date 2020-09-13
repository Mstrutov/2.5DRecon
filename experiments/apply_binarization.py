import cv2 as cv
import numpy as np


def get_mask(grayscale_texture, glob_threshold=10):
    # get the black background mask in order to afterwards transform the black background to
    # be in the same cluster in Otsu's binarization as the bright one
    threshold, black_background_mask = cv.threshold(grayscale_texture, glob_threshold, 255, cv.THRESH_BINARY_INV)

    # find the brightest pixel - it will be used as a bright background value
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(grayscale_texture)

    # making sure that grayscaled_texture (<glob_threshold) + 255 * coeff < 255
    coeff = np.min([maxVal / 255, 1 - glob_threshold / 255])

    # adding 255 * coeff to the black background to set it bright
    grayscale_texture = cv.addWeighted(grayscale_texture, 1, black_background_mask, coeff, 0)

    # apply otsu's binarization to the texture
    th, mask = cv.threshold(grayscale_texture, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # return the mask
    return mask


def remove_background(input_depthmap_path, input_texture_path, output_depthmap_path, glob_threshold=10):
    # get the images
    depth_map = cv.imread(input_depthmap_path, cv.IMREAD_GRAYSCALE)
    texture = cv.imread(input_texture_path, cv.IMREAD_COLOR)

    # preprocess the texture
    grayscale_texture = cv.cvtColor(texture, cv.COLOR_RGB2GRAY)

    mask = get_mask(grayscale_texture, glob_threshold)

    # apply mask to the depthmap and save the result
    masked_depth_map = cv.bitwise_and(depth_map, depth_map, mask=mask)
    cv.imwrite(output_depthmap_path, masked_depth_map)


# # datasets = ['2', '3', '4', '6', '7', '9', '12', '13', '14', '15', '17', '18', '20']
# datasets = ['HD. Bee wing new 30', 'HD. Bee wing new 100', 'HD. Bee wing new by-hand']
# # datasets = ['HD. Human hair 25',
# #             'HD. Human hair 50',
# #             'HD. Human hair 100',
# #             'HD. Human hair 200',
# #             'HD. Human hair 500',
# #             'HD. Human hair 1000',
# #             'HD. Human hair by-hand']
# dir = '../results'
# depthmap_method = 'FIJI.tif'
# texture_method = 'IJ.tif'
#
# for dataset_name in datasets:
#     input_texture = f'{dir}/{dataset_name}/Texture-{texture_method}'
#     input_depthmap = f'{dir}/{dataset_name}/Depthmap-{depthmap_method}'
#     output_depthmap = f'{dir}/{dataset_name}/Depthmap-OTSU_2.png'
#     remove_background(input_depthmap, input_texture, output_depthmap, glob_threshold=2)



