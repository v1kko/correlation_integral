"""
  Correlation Integral caluclations done fast with openmp
  Copyright (C) 2021  Victor Azizi - victor@lipsum.eu

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from numpy.distutils.core import Extension, setup
from sys import platform
from os import environ as env

ext1 = Extension(name = 'correlation_integral',
                 sources = ['correlation_integral.f90'],
                 extra_f90_compile_args = ['-fopenmp'],
                 libraries = [] if platform.startswith('win') else ['gomp'],
    )

if __name__ == "__main__":
  with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

  #Windows numpy distutils ignores every linker flag, so we force it
  if platform.startswith('win'):
    env['LDFLAGS']        = "-Xlinker --start-group -lgomp"

  setup(name              = "correlation_integral",
        version           = "0.0.4",
        url               = "https://github.com/v1kko/correlation_integral",
        description       = "Correlation Integral caluclations done fast with openmp",
        long_description  = long_description,
        author            = "Victor Azizi",
        author_email      = "victor@lipsum.eu",
        classifiers       = [
                            "Programming Language :: Python :: 3",
                            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                            "Operating System :: OS Independent",
                            ],
        python_requires   = ">=3.6",
        install_requires  = ["numpy>=1.18.0"],
        ext_modules       = [ext1],
        )
