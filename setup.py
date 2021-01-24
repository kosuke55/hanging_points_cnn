from __future__ import print_function

import distutils.spawn
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = "0.0.1"


if sys.argv[-1] == "release":
    if not distutils.spawn.find_executable("twine"):
        print(
            "Please install twine:\n\n\tpip install twine\n", file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        "git tag v{:s}".format(version),
        "git push origin master --tag",
        "python setup.py sdist",
        "twine upload dist/hanging_points_cnn-{:s}.tar.gz".format(
            version),

    ]
    for cmd in commands:
        print("+ {}".format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


setup_requires = []
install_requires = [
    'cameramodels',
    'gdown',
    'pybullet',
    'scikit-robot',
]


setup(
    name="hanging_points_cnn",
    version=version,
    description="hanging points cnn",
    author="kosuke55",
    author_email="kosuke.tnp@gmail.com",
    url="https://github.com/kosuke55/hanging_points_cnn",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts':
        ['visualize-function-points=hanging_points_cnn.apps.visualize_function_points:main',
         'infer-function-points=hanging_points_cnn.apps.infer_function_points:main',
         'unable-new-zipfile-serialization=hanging_points_cnn.apps.unable_new_zipfile_serialization:main'
         ]},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
