#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

PROJECT = 'DARIAH Topic Modelling'
VERSION = 'frozen'
AUTHOR = 'DARIAH-DE Wuerzburg Group'
AUTHOR_EMAIL = 'pielstroem@biozentrum.uni-wuerzburg.de'

setup(
    name='dariah_topics',
    version=VERSION,
    description=PROJECT,
    # url
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    # license
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6'
    ],
    # keywords
    packages=find_packages(exclude=['docs', 'demonstrator', 'grenzboten_sample', 'test', 'tutorial_supplementals']),
    install_requires=[
        'pandas>=0.19.2',
        'regex>=2017.01.14',
        'lda>=1.0.5',
        'numpy>=1.3',
        'bokeh>=0.12.6',
    ],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', PROJECT),
            'version': ('setup.py', VERSION),
            'release': ('setup.py', REVISION),
        }
    }
)
