# from https://github.com/tomrunia/OpticalFlow_Visualization
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    https://vision.middlebury.edu//flow/code/flow-code/colorcode.cpp
    '''

    # relative lengths of color transitions, based on perceptual similarity
    # (one can distinguish more shades between red and yellow than between yellow and green)
    # red to yellow
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # yellow to green
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # green to cyan
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # cyan to blue
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # blue to magenta
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # magenta to red
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    # initialize the output RGB image (or BGR)
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    # generate color wheel
    # colorwheel has shape [55*3]
    colorwheel = make_colorwheel()
    # number of colors in x and y
    num_colors = colorwheel.shape[0]

    # radius (magnitude)
    rad = np.sqrt(np.square(u) + np.square(v))
    # angle percentage (pi is 100%)
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1.0) / 2.0 * (num_colors - 1.0)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % num_colors
    # k1[k1 == ncols] = 0
    f = fk - k0
    # f = 0 # uncomment to see original color wheel

    # assign RGB values
    for i in range(colorwheel.shape[1]):
        # all color candidates
        tmp = colorwheel[:, i]
        color_0 = tmp[k0] / 255.0
        color_1 = tmp[k1] / 255.0
        final_color = (1 - f)*color_0 + f * color_1

        # increase saturation with radius
        # if rad <= 1:
        #     final_color = 1 - rad * (1 - final_color)
        # else:
        #     final_color *= 0.75
        idx = (rad <= 1)
        final_color[idx]  = 1 - rad[idx] * (1 - final_color[idx])
        final_color[~idx] = final_color[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        if convert_to_bgr:
            color_idx = 2-i
        else:
            color_idx = i

        flow_image[:, :, color_idx] = np.floor(255 * final_color)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    # horizontal and vertical velocities
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    # compute velocity magnitude
    vel_magnitude = np.sqrt(np.square(u) + np.square(v))
    max_vel = np.max(vel_magnitude)

    # normalize velocity
    epsilon = 1e-5
    u = u / (max_vel + epsilon)
    v = v / (max_vel + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)