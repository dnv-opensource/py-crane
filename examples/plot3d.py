"""
Note1 View angle interactive changes are implemented! Press mouse button and move
Note2 that plt.figure.canvas.mpl_connect( event, callback) enables more interactive functions with the plot.
"""

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 9))
# line = Line3D( [1,2], [3,4], [5,6])
ax = plt.axes(projection="3d")  # , data=line)
dat = ax.plot([1, 2], [3, 4], [5, 6], linewidth=5)
# print( dat, dat[0].get_data_3d()) # dat is a collection of 3d lines!

ax.view_init(-140, 60)  # type: ignore ## according to matplotlib documentation
for x1 in range(10):
    dat[0].set_data_3d([x1, 2], [3, 4], [5, 6])  # type: ignore ## according to matplotlib documentation
    # print("x1", x1)
    plt.pause(1)

plt.show()
