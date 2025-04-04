Introduction
============
The package extends the `component_model` package to provide the means to construct cranes.
The package can be used to define concrete cranes and simulate their usage, see `tests/resources/mobile-crane.py` as example.

Getting Started
---------------
A concrete crane model should consist of an extension of the Crane object,

* defining general model variables like `name`, `author`, `version`, etc.
* defining the anticipated set of booms
* if there is a wish to keep parameters configurable, these can be kept as parameters of the extended model.
* In a separate script, instantiate the concrete crane, test it and run `Model.build(...)` to make the FMU.

See the file `tests/resources/mobile_crane.py` and related test files
as an example on how a concrete crane can be defined and used.


Installation
------------

``pip install crane-fmu``


Usage example
-------------
This is a simple mobile crane, like they are used on e.g. building sites with

* a short pedestal, which can be turn around,
* one boom with variable length and which can be lifted
* one rope, where loads can be attached.

.. code-block:: Python

    from crane_fmu.crane import Crane

    class MobileCrane(Crane):
        """Simple mobile crane for FMU testing purposes.
        The crane has a short pedestal, one variable-length stiff boom and a rope.
        The size and weight of the various parts can be configured.

        Args:
            name (str) : name of the crane type
            description (str) : short description
            author (str)
            version (str)
            pedestalMass (str) : mass of the pedestal - quantity and unit as string
            pedestalHeight (str) : height (fixed) of the pedestal, with units
            boomMass (str) : mass of the single boom, with units
            boomLength0 (str) : minimum length of the boom, with units
            boomLength1 (str) : maximum length of the boom, with units
        """

        def __init__(
            self,
            name: str = "mobile_crane",
            description: str = "Simple mobile crane (for FMU testing) with short pedestal, one variable-length elevation boom and a rope",
            author: str = "DNV, SEACo project",
            version: str = "0.2",
            pedestalMass: str = "10000.0 kg",
            pedestalHeight: str = "3.0 m",
            boomMass: str = "1000.0 kg",
            boomLength0: str = "8 m",
            boomLength1: str = "50 m",
            rope_mass_range: tuple = ("50kg", "2000 kg"),
            **kwargs,
        ):
            super().__init__(name=name, description=description, author=author, version=version, **kwargs)
            _ = self.add_boom(
                name="pedestal",
                description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
                mass=pedestalMass,
                massCenter=(0.5, -1.0, 0.8),
                boom=(pedestalHeight, "0deg", "0deg"),
                boom_rng=(None, None, ("0deg", "360deg")),
            )
            _ = self.add_boom(
                name="boom",
                description="The boom. Can be lifted and length can change within the given range",
                mass=boomMass,
                massCenter=(0.5, 0, 0),
                boom=(boomLength0, "90deg", "0deg"),
                boom_rng=((boomLength0, boomLength1), (0, "90deg"), None),
            )
            _ = self.add_boom(
                name="rope",
                description="The rope fixed to the last boom. Flexible connection",
                mass="50.0 kg",  # so far basically the hook
                massCenter=0.95,
                mass_rng=rope_mass_range,
                boom=("1e-6 m", "180deg", "0 deg"),
                boom_rng=(
                    ("1e-6 m", boomLength1),
                    ("90deg", "270deg"),
                    ("-180deg", "180deg"),
                ),
                damping=50.0,
                animationLW=2,
            )
            # make sure that _comSub is calculated for all booms:
            self.calc_statics_dynamics(None)

        def do_step(self, currentTime, stepSize):
            status = super().do_step(currentTime, stepSize)
            # print(f"Time {currentTime}, {self.rope_tip}")
            # print(f"MobileCrane.do_step. Status {status}")
            return status



Testing and usage of the `MobileCrane.fmu` is demonstrated as part of related `test_***` files.


Development Setup
-----------------

1. Install uv
^^^^^^^^^^^^^
This project uses `uv` as package manager.

If you haven't already, install `uv <https://docs.astral.sh/uv/>`_, preferably using it's `"Standalone installer" <https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2/>`_ method:

..on Windows:

``powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"``

..on MacOS and Linux:

``curl -LsSf https://astral.sh/uv/install.sh | sh``

(see `docs.astral.sh/uv <https://docs.astral.sh/uv/getting-started/installation//>`_ for all / alternative installation methods.)

Once installed, you can update `uv` to its latest version, anytime, by running:

``uv self update``

2. Install Python
^^^^^^^^^^^^^^^^^
This project requires Python 3.10 or later.

If you don't already have a compatible version installed on your machine, the probably most comfortable way to install Python is through ``uv``:

``uv python install``

This will install the latest stable version of Python into the uv Python directory, i.e. as a uv-managed version of Python.

Alternatively, and if you want a standalone version of Python on your machine, you can install Python either via ``winget``:

``winget install --id Python.Python``

or you can download and install Python from the `python.org <https://www.python.org/downloads//>`_ website.

3. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^
Clone the crane-fmu repository into your local development directory:

``git clone https://github.com/dnv-opensource/crane-fmu path/to/your/dev/crane-fmu``

4. Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^
Run ``uv sync`` to create a virtual environment and install all project dependencies into it:

``uv sync``

5. (Optional) Activate the virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When using ``uv``, there is in almost all cases no longer a need to manually activate the virtual environment.

``uv`` will find the ``.venv`` virtual environment in the working directory or any parent directory, and activate it on the fly whenever you run a command via `uv` inside your project folder structure:

``uv run <command>``

However, you still *can* manually activate the virtual environment if needed.
When developing in an IDE, for instance, this can in some cases be necessary depending on your IDE settings.
To manually activate the virtual environment, run one of the "known" legacy commands:

..on Windows:

``.venv\Scripts\activate.bat``

..on Linux:

``source .venv/bin/activate``

6. Install pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``.pre-commit-config.yaml`` file in the project root directory contains a configuration for pre-commit hooks.
To install the pre-commit hooks defined therein in your local git repository, run:

``uv run pre-commit install``

All pre-commit hooks configured in ``.pre-commit-config.yam`` will now run each time you commit changes.

7. Test that the installation works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To test that the installation works, run pytest in the project root folder:

``uv run pytest``


Meta
----
Copyright (c) 2024 `DNV <https://www.dnv.com/>`_ AS. All rights reserved.

Siegfried Eisinger - siegfried.eisinger@dnv.com

Distributed under the MIT license. See `LICENSE <LICENSE.md/>`_ for more information.

`https://github.com/dnv-opensource/crane-fmu <https://github.com/dnv-opensource/crane-fmu/>`_

Contribute
----------
Anybody in the FMU, OSPand SEACo community is especially welcome to contribute to this code, to make it better,
and especially including other features from model assurance and from SEACo issues.

To contribute, follow these steps:

1. Fork it `<https://github.com/dnv-opensource/crane-fmu/fork/>`_
2. Create an issue in your GitHub repo
3. Create your branch based on the issue number and type (``git checkout -b issue-name``)
4. Evaluate and stage the changes you want to commit (``git add -i``)
5. Commit your changes (``git commit -am 'place a descriptive commit message here'``)
6. Push to the branch (``git push origin issue-name``)
7. Create a new Pull Request in GitHub

For your contribution, please make sure you follow the `STYLEGUIDE <STYLEGUIDE.md/>`_ before creating the Pull Request.


