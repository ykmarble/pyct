#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
from operator import add
import shutil

def main():
    LOGDIR = "iterout"
    THRESOLD = 90

    dryrun = True
    if len(sys.argv) == 2 and sys.argv[1] == "1":
        dryrun = False
    else:
        print "dryrun mode"


    dirs = [p for p in (os.path.join(LOGDIR, d) for d in os.listdir(LOGDIR)) if os.path.isdir(p)]
    while len(dirs) != 0:
        path = dirs.pop()
        content = os.listdir(path)
        delete_flag = False

        # empty dir
        if len(content) == 0:
            delete_flag = True

        # log dir
        elif "iterlog.txt" in content:
            with open(os.path.join(path, "iterlog.txt")) as f:
                nlines = reduce(add, (1 for i in f.xreadlines()), 0)
                if nlines < THRESOLD:
                    delete_flag = True

        # dig sub dir
        else:
            for c in content:
                subpath = os.path.join(path, c)
                if os.path.isdir(subpath):
                    dirs.append(subpath)

        if delete_flag:
            print "delete {}".format(path), content
            if not dryrun:
                shutil.rmtree(path)


if __name__ == '__main__':
    main()
