#!/usr/bin/env python3

from pyct import utils
import numpy
import skimage.transform
from math import sin, cos, pi


def build_chord_encoder(origin, theta=0, flip_y=False):
    orig_x, orig_y = origin
    sin_t = sin(-theta)
    cos_t = cos(-theta)
    def chord_encode(pos):
        x, y = pos
        x -= orig_x
        y -= orig_y
        if flip_y:
            y = -y
        rx = cos_t * x - sin_t * y
        ry = sin_t * x + cos_t * y
        return (rx, ry)
    return chord_encode


def draw_elipse(canvas, level, pos, r, theta=0):
    offset_x, offset_y = canvas.shape
    offset_x = (offset_x - 1) / 2.
    offset_y = (offset_y - 1) / 2.
    main_chord = build_chord_encoder((offset_x, offset_y), flip_y=True)
    elipse_chord = build_chord_encoder(pos, theta=theta)
    rx, ry = r
    rx_2 = rx ** 2
    ry_2 = ry ** 2

    if not hasattr(level, "__call__"):
        v = level
        level = lambda x, y: v

    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            je, ie = elipse_chord(main_chord((j, i)))
            if (ry_2 * je**2 + rx_2 * ie**2 < rx_2 * ry_2):
                canvas[i, j] += level(je, ie)

def plySheppLoganPhantom(NoSP=256):
    canvas = SheppLoganPhantom(NoSP)
    scale = (NoSP - 1) / 2.

    #b_level = lambda x, y: -0.01 * y / (0.874 * scale)
    #draw_elipse(canvas, b_level,(  0.   * scale, -0.0184 * scale), ( 0.6624 * scale, 0.874 * scale),   0            )  # ply b

    c_level = lambda x, y: -0.02 * y / (0.31 * scale)
    draw_elipse(canvas, c_level,(  0.22 * scale,  0.     * scale), ( 0.11   * scale, 0.31  * scale), -18 / 180. * pi)  # ply c

    d_level = lambda x, y: -0.02 * y / (0.41 * scale)
    draw_elipse(canvas, d_level,( -0.22 * scale,  0.     * scale), ( 0.16   * scale, 0.41  * scale),  18 / 180. * pi)  # ply d

    #e_level = lambda x, y: -0.02 *(1 - y / (0.25 * scale))
    #draw_elipse(canvas, e_level,(  0.   * scale,  0.35   * scale), ( 0.21   * scale, 0.25  * scale),   0            )  # ply e

    f_level = lambda x, y: -0.01 * y / (0.046 * scale)
    draw_elipse(canvas, f_level,(  0.   * scale,  0.1    * scale), ( 0.046  * scale, 0.046 * scale),   0            )  # ply f

    return canvas

def SheppLoganPhantom(NoSP=256):
    canvas = numpy.zeros((NoSP, NoSP))
    scale = (NoSP - 1) / 2.

    # the definition of 2x2 Shepp-Logan phantom (from Wikipedia)
    draw_elipse(canvas,  2   ,(  0.   * scale,  0.     * scale), ( 0.69   * scale, 0.92  * scale),   0            )  # a
    draw_elipse(canvas, -0.98,(  0.   * scale, -0.0184 * scale), ( 0.6624 * scale, 0.874 * scale),   0            )  # b
    draw_elipse(canvas, -0.02,(  0.22 * scale,  0.     * scale), ( 0.11   * scale, 0.31  * scale), -18 / 180. * pi)  # c
    draw_elipse(canvas, -0.02,( -0.22 * scale,  0.     * scale), ( 0.16   * scale, 0.41  * scale),  18 / 180. * pi)  # d
    draw_elipse(canvas,  0.01,(  0.   * scale,  0.35   * scale), ( 0.21   * scale, 0.25  * scale),   0            )  # e
    draw_elipse(canvas,  0.01,(  0.   * scale,  0.1    * scale), ( 0.046  * scale, 0.046 * scale),   0            )  # f
    draw_elipse(canvas,  0.01,(  0.   * scale, -0.1    * scale), ( 0.046  * scale, 0.046 * scale),   0            )  # g
    draw_elipse(canvas,  0.01,( -0.08 * scale, -0.605  * scale), ( 0.046  * scale, 0.023 * scale),   0            )  # h
    draw_elipse(canvas,  0.01,(  0.   * scale, -0.605  * scale), ( 0.023  * scale, 0.023 * scale),   0            )  # i
    draw_elipse(canvas,  0.01,(  0.06 * scale, -0.605  * scale), ( 0.023  * scale, 0.046 * scale),   0            )  # j

    return canvas

def modSheppLoganPhantom(NoSP=256):
    canvas = numpy.zeros((NoSP, NoSP))
    scale = (NoSP - 1) / 2.

    # the definition of 2x2 Shepp-Logan phantom (from Wikipedia)
    draw_elipse(canvas,  1  ,(  0.   * scale,  0.     * scale), ( 0.69   * scale, 0.92  * scale),   0             )  # a
    draw_elipse(canvas, -0.8,(  0.   * scale, -0.0184 * scale), ( 0.6624 * scale, 0.874 * scale),    0            )  # b
    draw_elipse(canvas, -0.2,(  0.22 * scale,  0.     * scale), ( 0.11   * scale, 0.31  * scale),  -18 / 180. * pi)  # c
    draw_elipse(canvas, -0.2,( -0.22 * scale,  0.     * scale), ( 0.16   * scale, 0.41  * scale),   18 / 180. * pi)  # d
    draw_elipse(canvas,  0.1,(  0.   * scale,  0.35   * scale), ( 0.21   * scale, 0.25  * scale),    0            )  # e
    draw_elipse(canvas,  0.1,(  0.   * scale,  0.1    * scale), ( 0.046  * scale, 0.046 * scale),    0            )  # f
    draw_elipse(canvas,  0.1,(  0.   * scale, -0.1    * scale), ( 0.046  * scale, 0.046 * scale),    0            )  # g
    draw_elipse(canvas,  0.1,( -0.08 * scale, -0.605  * scale), ( 0.046  * scale, 0.023 * scale),    0            )  # h
    draw_elipse(canvas,  0.1,(  0.   * scale, -0.606  * scale), ( 0.023  * scale, 0.023 * scale),    0            )  # i
    draw_elipse(canvas,  0.1,(  0.06 * scale, -0.606  * scale), ( 0.023  * scale, 0.046 * scale),    0            )  # j
    draw_elipse(canvas, -0.2,(  0.5  * scale, -0.52   * scale), ( 0.2    * scale, 0.04  * scale), 60.5 / 180. * pi)  # k

    return canvas

def simplePhantom(NoSP=256):
    canvas = numpy.zeros((NoSP, NoSP))
    scale = (NoSP - 1) / 2.

    draw_elipse(canvas, 0.5, (0 * scale, 0 * scale), (0.8 * scale, 0.8 * scale), 0)
    draw_elipse(canvas, 0.2, (0.3 * scale, 0.3 * scale), (0.08 * scale, 0.08 * scale), 0)
    draw_elipse(canvas, 0.2, (-0.3 * scale, 0 * scale), (0.1 * scale, 0.1 * scale), 0)

    return canvas

def main():
    output_length = 256
    oversample_scale = 2.5

    oversampled_phantom = simplePhantom(int(output_length * oversample_scale))
    phantom = skimage.transform.resize(oversampled_phantom, (output_length, output_length))

    utils.save_rawimage(phantom, "modSLP.dat")

if __name__ == '__main__':
    main()
