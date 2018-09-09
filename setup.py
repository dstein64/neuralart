import os
from setuptools import setup

version_txt = os.path.join(os.path.dirname(__file__), 'neuralart', 'version.txt')
with open(version_txt, 'r') as f:
    version = f.read().strip()

setup(
    name='neuralart',
    packages=['neuralart'],
    package_data={'neuralart': ['version.txt']},
    scripts=['bin/neuralart'],
    license='MIT',
    version=version,
    description='A library for rendering generative art from a randomly initialized neural network.',
    long_description=open('README.rst').read(),
    author='Daniel Steinberg',
    author_email='ds@dannyadam.com',
    url='https://github.com/dstein64/neuralart',
    keywords=['neural-networks', 'generative-art'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Artistic Software',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=['pillow', 'torch']
)
