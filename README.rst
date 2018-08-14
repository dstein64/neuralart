A library and command line utility for rendering generative art from a randomly
initialized neural network.

Inspired by the following blog posts and pages on `otoro.net <http://otoro.net/>`__
- `Neural Network Generative Art in Javascript <http://blog.otoro.net/2015/06/19/neural-network-generative-art/>`__
- `Generating Abstract Patterns with TensorFlow <http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/>`__
- `Neurogram <http://blog.otoro.net/2015/07/31/neurogram/`__
- `Interactive Neural Network Art <http://otoro.net/ml/netart/`__

Requirements
------------

*neuralart* supports Python 2.7 and Python 3.x.

Linux, Mac, and Windows are supported.

Other operating systems may be compatible if the dependencies can be properly installed.

Dependencies
~~~~~~~~~~~~

- NumPy
- SciPy
- pillow

Installation
------------

`neuralart <https://pypi.python.org/pypi/neuralart>`__ is available on PyPI,
the Python Package Index.

::

    $ pip install neuralart

Command Line Utility
--------------------

There is a command line utility for generating images. Use the :code:`--help`
flag for more information.

::

    $ neuralart --help

Library Example Usage
---------------------

See `example.py <https://github.com/dstein64/neuralart/blob/master/example.py>`__.

License
-------

The code in this repository has an `MIT License <https://en.wikipedia.org/wiki/MIT_License>`__.

See `LICENSE <https://github.com/dstein64/neuralart/blob/master/LICENSE>`__.