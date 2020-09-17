import cv2
import numpy


def focus_functions_list():
    return ['LAPV', 'LAPM', 'TENG', 'MLOG', 'VOLL4']


def name_to_function(name):
    switcher = {
        'LAPV': LAPV,
        'LAPM': LAPM,
        'TENG': TENG,
        'MLOG': MLOG,
        'VOLL4': VOLL4
    }

    return switcher.get(name)


# Variance of Laplacian
def LAPV(img):
    return numpy.std(cv2.Laplacian(img, cv2.CV_64F)) ** 2


# Modified Laplacian
def LAPM(img):
    kernel = numpy.array([-1, 2, -1])
    laplacianX = numpy.abs(cv2.filter2D(img, -1, kernel))
    laplacianY = numpy.abs(cv2.filter2D(img, -1, kernel.T))
    return numpy.mean(laplacianX + laplacianY)


# Tenengrad focus measure operator
def TENG(img):
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    return numpy.mean(gaussianX * gaussianX + gaussianY * gaussianY)


def default_parts(img, xseg_size, yseg_size, func_name):
    if func_name == 'TENG':
        return TENG_parts(img, xseg_size, yseg_size)

    (height, width, _) = img.shape

    xseg_amount = width // xseg_size + (1 if width % xseg_size != 0 else 0)
    yseg_amount = height // yseg_size + (1 if height % yseg_size != 0 else 0)
    func_res = numpy.zeros((yseg_amount, xseg_amount))
    func = name_to_function(func_name)

    yseg_cnt = 0
    for lu_y in range(0, height, yseg_size):
        xseg_cnt = 0
        for lu_x in range(0, width, xseg_size):
            ru_x = lu_x + xseg_size if lu_x + xseg_size <= width else width
            ld_y = lu_y + yseg_size if lu_y + yseg_size <= height else height

            image_sector = img[lu_y:ld_y, lu_x:ru_x].copy()
            func_res[yseg_cnt][xseg_cnt] = func(image_sector)

            xseg_cnt += 1

        yseg_cnt += 1
    return func_res


def TENG_parts(img, xseg_size, yseg_size):
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    (height, width, _) = img.shape

    xseg_amount = width // xseg_size + (1 if width % xseg_size != 0 else 0)
    yseg_amount = height // yseg_size + (1 if height % yseg_size != 0 else 0)
    func_res = numpy.zeros((yseg_amount, xseg_amount))

    yseg_cnt = 0
    for lu_y in range(0, height, yseg_size):
        xseg_cnt = 0
        for lu_x in range(0, width, xseg_size):
            ru_x = lu_x + xseg_size if lu_x + xseg_size <= width else width
            ld_y = lu_y + yseg_size if lu_y + yseg_size <= height else height

            func_res[yseg_cnt][xseg_cnt] = numpy.mean(
                gaussianX[lu_y:ld_y, lu_x:ru_x] * gaussianX[lu_y:ld_y, lu_x:ru_x] +
                gaussianY[lu_y:ld_y, lu_x:ru_x] * gaussianY[lu_y:ld_y, lu_x:ru_x]
            )

            xseg_cnt += 1

        yseg_cnt += 1
    return func_res


# MLOG focus measure operator
def MLOG(img):
    return numpy.max(cv2.convertScaleAbs(cv2.Laplacian(img, 3)))


# Vollath-4 operator to measure focus
def VOLL4(gray_img):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(int)

    tmp_image = numpy.zeros(gray_img.shape, dtype=int)
    tmp_image[:, :tmp_image.shape[1] - 1] = gray_img[:, 1:]
    sum1 = numpy.multiply(gray_img, tmp_image).sum()

    tmp_image = numpy.zeros(gray_img.shape, dtype=int)
    tmp_image[:, :tmp_image.shape[1] - 2] = gray_img[:, 2:]
    sum2 = numpy.multiply(gray_img, tmp_image).sum()

    return sum1 - sum2
