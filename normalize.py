#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import utils
import numpy
import sys
import os


def main():
    if len(sys.argv) == 1:
        print "Usage: {} <file>".format(sys.argv[0])
        sys.exit(1)

    for path in sys.argv[1:]:
        dname, fname, ext = utils.decompose_path(path)

        img = utils.load_rawimage(path)

        if img is None or img.shape[0] != img.shape[1]:
            print "{} is invalid".format(path)

        mask = numpy.zeros_like(img)
        c = (img.shape[0] - 1) / 2.
        a = b = c
        utils.crop_elipse(mask, (c, c), a+0, b+0, value=1)
        utils.crop_elipse(mask, (c, c), a-2.5, b-2.5, value=0)
        mask[20:] = 0
        bg = numpy.average(img[mask == 1])
        img -= bg

        mask[:, :] = 0
        utils.crop_elipse(mask, (c, c), a-0.5, b-0.5, value=1)
        img[mask == 0] = 0

#        img[-20:] = 0

        img[img < 0] = 0
        utils.normalize(img)

        out_path = os.path.join(dname, "normalized_{}.{}".format(fname, ext))
        utils.save_rawimage(img, out_path)
        print "saved normalized image to {}".format(out_path)


if __name__ == '__main__':
    main()
