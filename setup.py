# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup script for fkeras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import setuptools

with io.open("README.md", "r", encoding="utf8") as fh:
  long_description = fh.read()

setuptools.setup(
    name="FKeras",
    version="0.0.0",
    author="Olivia Weng, Andres Meza",
    author_email="oweng@ucsd.edu, anmeza@ucsd.edu",
    maintainer="Olivia Weng, Andres Meza",
    maintainer_email="oweng@ucsd.edu, anmeza@ucsd.edu",
    packages=setuptools.find_packages(),
    scripts=[],
    url="",
    license="Apache v.2.0",
    description="Fault tolerance package for Keras & QKeras",
    long_description=long_description,
    install_requires=[
        "numpy>=1.16.0",
        "scipy>=1.4.1",
        "pyparser",
        "setuptools>=41.0.0",
        "tensorflow-model-optimization>=0.2.1",
        "networkx>=2.1",
        "keras-tuner>=1.0.1",
        "scikit-learn>=0.23.1",
        "tqdm>=4.48.0",
        "qkeras>=0.9.0"
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
)