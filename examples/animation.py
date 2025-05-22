import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection="3d")


def gen(n):
    phi = 0
    while phi < 2 * np.pi:
        yield np.array([np.cos(phi), np.sin(phi), phi])
        phi += 2 * np.pi / n


def update(num, data, line):
    print("UPDATE", num, data[:2, :num])
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])


N = 100
data = np.array(list(gen(N))).T
(line,) = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

# Setting the axes properties
ax.set_xlim((-1.0, 1.0))
ax.set_xlabel("X")

ax.set_ylim((-1.0, 1.0))
ax.set_ylabel("Y")

ax.set_zlim((0.0, 10.0))  # type: ignore  ## according to matplotlib documentation
ax.set_zlabel("Z")  # type: ignore  ## according to matplotlib documentation

print("Before animation.", N, len(data))

ani = animation.FuncAnimation(
    fig,
    update,  # type: ignore  ## return value of update() can be omitted if blit=False!
    frames=N,
    fargs=(data, line),
    interval=10000 / N,
    blit=False,
)
explanation = """
class matplotlib.animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None, save_count=None, *, cache_frame_data=True, **kwargs)
`FuncAnimation <https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_

fig:
   The figure object used to get needed events, such as draw or resize.
func=update:
   The function to call at each frame. Signature: def func(frame, *fargs) -> iterable_of_artists
frames (iterable, int, generator function, or None)
   Source of data to pass func and each frame of the animation

   int:
      equivalent to passing range(frames)

init_func (callable, None):
   Optional dedicated initialization of the figure. Signature: def init_func() -> iterable_of_artists
   Needed if blit=True
fargs (tuple, None):
   Additional arguments to pass to each call to func (alternative: use functools.partial)
interval (int)=200
   Delay between frames in milliseconds.
blit (bool)=False:
   Whether blitting is used to optimize drawing, i.e. several bitmaps are combined into one using a boolean function.
"""
# ani.save('matplot003.gif', writer='imagemagick')
plt.show()
