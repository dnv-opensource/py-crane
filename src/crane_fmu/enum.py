from enum import Flag


class Change(Flag):
    """Fag for changes in :py:class:`.Boom` length or rotational changes.
    Used when trasnmitting changes within one boom to its children.
    """

    POS = 1
    ROT = 2
    ALL = POS | ROT
