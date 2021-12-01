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

from numpy.distutils.core import Extension

ext1 = Extension(name = 'correlation_integral',
                 sources = ['correlation_integral.f90'],
                 libraries = ['gomp'],
    )

if __name__ == "__main__":
  with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
  from numpy.distutils.core import setup
  from os import environ as env
  env['F90FLAGS']='-fopenmp'
  setup(name              = "correlation_integral",
        version           = "0.0.1",
        url               = "https://github.com/v1kko/correlation_integral",
        description       = "Correlation Integral caluclations done fast with openmp",
        long_description  = long_description,
        author            = "Victor Azizi",
        author_email      = "victor@lipsum.eu",
        classifiers       = [
                            "Programming Language :: Python :: 3",
                            "License :: OSI Approved :: GPLv3 License",
                            "Operating System :: OS Independent",
                            ],
        python_requires   = ">=3.6",
        ext_modules       = [ext1],
        )
