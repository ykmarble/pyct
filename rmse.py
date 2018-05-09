#!/usr/bin/env python2
#-*- coding: utf-8 -*-

import utils
from pylab import *

def main():
    import sys
    for path in sys.argv[1:]:
        data = loadtxt(path)
        (_, label, _) = utils.decompose_path(path)
        plot(data[:, 1], label=label)
    legend()
    show()

if __name__ == "__main__":
    main()
