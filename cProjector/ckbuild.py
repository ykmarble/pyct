#!/usr/bin/env python3

import pathlib

def main():
    lib = pathlib.Path("./sysmat_cpp.so")
    src = pathlib.Path("./src").glob("*")
    libisold = False
    libmt = lib.stat().st_mtime
    for s in src:
        libisold = s.stat().st_mtime > libmt
        if libisold: break
    if libisold:
        print("Library need to be rebuilt.")
    else:
        print("Library is latest.")


if __name__ == "__main__":
    main()

