.. image:: https://img.shields.io/pypi/v/skeleton.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/skeleton.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/skeleton

.. image:: https://github.com/jaraco/skeleton/workflows/tests/badge.svg
   :target: https://github.com/jaraco/skeleton/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. .. image:: https://readthedocs.org/projects/skeleton/badge/?version=latest
..    :target: https://skeleton.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2022-informational
   :target: https://blog.jaraco.com/skeleton


**Pupil Labs IMU Vizualization Tool**

----------------

**Installation**

Tested with Python 3.10. Setup new environment using conda. Activate environment. Then run the following command within the repository:

```pip install -e .```

Then you should be good to go. 

**Usage**

To start the real-time visualization, in the terminal run:

```plimu_viz --address [IP OF COMPANION PHONE] --port [PORT TO USE] --show_stars [BOOLEAN]```

The second and third arguments are OPTIONAL with defaults port=8080 and show_stars=False. 

Depending on firmware version of your module, you might need to start the scene camera preview in the Companion App to trigger streaming of the IMU. 


