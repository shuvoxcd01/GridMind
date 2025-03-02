"""
This module is based on the tile coding implementation from ```Tile Coding Software -- Reference Manual, Version 3.0``` by Rich S. Sutton (See: http://incompleteideas.net/tiles/tiles3.html).
"""

from math import floor, log
from itertools import zip_longest

import numpy as np

basehash = hash


class IHT:
    "Structure to handle collisions"

    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return (
            "Collision table:"
            + " size:"
            + str(self.size)
            + " overfullCount:"
            + str(self.overfullCount)
            + " dictionary:"
            + str(len(self.dictionary))
            + " items"
        )

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount == 0:
                print("IHT full, starting to allow collisions")
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count


def hashcoords(coordinates, m, readonly=False):
    if type(m) == IHT:
        return m.getindex(tuple(coordinates), readonly)
    if type(m) == int:
        return basehash(tuple(coordinates)) % m
    if m == None:
        return coordinates


class TileCoding:
    def __init__(self, ihtORsize: int = 4096, numtilings: int = 8):
        self.ihtORsize = ihtORsize
        self.numtilings = numtilings

    @staticmethod
    def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
        """returns num-tilings tile indices corresponding to the floats and ints"""
        qfloats = [floor(f * numtilings) for f in floats]
        Tiles = []
        for tiling in range(numtilings):
            tilingX2 = tiling * 2
            coords = [tiling]
            b = tiling
            for q in qfloats:
                coords.append((q + b) // numtilings)
                b += tilingX2
            coords.extend(ints)
            Tiles.append(hashcoords(coords, ihtORsize, readonly))
        return Tiles

    @staticmethod
    def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
        """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
        qfloats = [floor(f * numtilings) for f in floats]
        Tiles = []
        for tiling in range(numtilings):
            tilingX2 = tiling * 2
            coords = [tiling]
            b = tiling
            for q, width in zip_longest(qfloats, wrapwidths):
                c = (q + b % numtilings) // numtilings
                coords.append(c % width if width else c)
                b += tilingX2
            coords.extend(ints)
            Tiles.append(hashcoords(coords, ihtORsize, readonly))
        return Tiles

    def __call__(self, floats, ints=[], readonly=False):
        tiles = self.tiles(self.ihtORsize, self.numtilings, floats, ints, readonly)
        tiles = np.array(tiles)
        return tiles


if __name__ == "__main__":
    iht = IHT(4096)
    tc = TileCoding(iht, 8)
    tiles = tc([0.1, 0.1])
    print(tiles)
    x = -0.5675576
    xdot = 0.0
    A = [1]
    print(tc.tiles(iht, 8, [8 * x / (0.5 + 1.2), 8 * xdot / (0.07 + 0.07)], A))
