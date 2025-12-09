class Animation:
    """Animation of the crane via matplotlib.
    Due to issues with multiple CPU processes, this can currently not be used in conjunction with OSP.

    Args:
        crane (Crane): a reference to the crane which shall be animated
        elements (dict)={}: a dict of visual elements to include in the animation.
          Each dictionary element is represented by an empty list which is filled by the element references during init,
          so that their position, ... can be changed during the animation
        interval (float)=0.1: waiting interval between simulation steps in s
        viewAngle (tuple)=(20,45,0): Optional change of initial view angle as (elevation, azimuth, roll) (in degrees)
    """

    def __init__(
        self,
        crane: Crane,
        elements: dict[str, list] | None = None,
        interval: float = 0.1,
        figsize: tuple[float, float] = (9, 9),
        xlim: tuple[float, float] = (-10, 10),
        ylim: tuple[float, float] = (-10, 10),
        zlim: tuple[float, float] = (0, 10),
        viewAngle: tuple[float, float, float] = (20, 45, 0),
    ):
        """Perform the needed initializations of an animation."""
        self.crane: Crane = crane
        self.elements: dict[str, Any] | None = elements
        self.interval: float = interval

        _ = plt.ion()  # run the GUI event loop
        self.fig: Figure = plt.figure(figsize=figsize, layout="constrained")
        ax: Axes3D = Axes3D(fig=self.fig)
        #        ax = plt.axes(projection="3d")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=viewAngle[0], azim=viewAngle[1], roll=viewAngle[2])
        sub: list[list] = [[], [], []]
        if isinstance(self.elements, dict):
            for b in self.crane.booms():  # walk along the series of booms
                if "booms" in self.elements:  # draw booms
                    self.elements["booms"].append(
                        ax.plot(
                            [b.origin[0], b.end[0]],
                            [b.origin[1], b.end[1]],
                            [b.origin[2], b.end[2]],
                            linewidth=b.animationLW,
                        )
                    )
                if "c_m" in self.elements:  # write mass of boom as string on center of mass point
                    self.elements["c_m"].append(
                        ax.text(
                            b.c_m[0],
                            b.c_m[1],
                            b.c_m[2],
                            s=str(int(b.mass.start)),  # type: ignore ## pyright confusion about 3D plots
                            color="black",
                        )
                    )
                if "c_m_sub" in self.elements:
                    for i in range(3):
                        sub[i].append(b.c_m_sub[1][i])
            if "c_m_sub" in self.elements and len(sub[0]):
                self.elements["c_m_sub"].append(
                    ax.plot(sub[0], sub[1], sub[2], marker="*", color="red", linestyle="")
                )  # need to put them in as plot and not scatter3d, such that coordinates can be changed in a good way
            if "current_time" in self.elements:
                self.elements["current_time"].append(
                    ax.text(
                        ax.get_xlim()[0],
                        ax.get_ylim()[0],
                        ax.get_zlim()[0],
                        s="time=0",
                        color="blue",
                    )
                )

    def update(self, current_time=None):
        """Based on the updated crane, update data as defined in elements."""
        sub: list[list] = [[], [], []]
        assert isinstance(self.elements, dict), "elements dict required at this stage"
        for i, b in enumerate(self.crane.booms()):
            if "booms" in self.elements:
                assert self.elements["booms"] is not None
                self.elements["booms"][i][0].set_data_3d(
                    [b.origin[0], b.end[0]],
                    [b.origin[1], b.end[1]],
                    [b.origin[2], b.end[2]],
                )
            if "c_m" in self.elements:
                assert self.elements["c_m"] is not None
                self.elements["c_m"][i].set_x(b.c_m_sub[1][0])
                self.elements["c_m"][i].set_y(b.c_m_sub[1][1])
                self.elements["c_m"][i].set_z(b.c_m_sub[1][2])
            if "c_m_sub" in self.elements:
                for i in range(3):
                    sub[i].append(b.c_m_sub[1][i])
        if "c_m_sub" in self.elements and len(sub[0]):
            self.elements["c_m_sub"][0][0].set_data_3d(sub[0], sub[1], sub[2])
        if "current_time" in self.elements and current_time is not None:
            self.elements["current_time"][0].set_text("time=" + str(round(current_time, 1)))

        self.fig.canvas.draw_idle()  # drawing updated values
        self.fig.canvas.flush_events()  # This will run the GUI event loop until all UI events currently waiting have been processed
        # time.sleep( self.interval)

    def interactive_off(self):
        plt.ioff()
