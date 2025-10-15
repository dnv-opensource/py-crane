from enum import Flag


class Change(Flag):
    POS = 1
    ROT = 2
    ALL = POS | ROT
