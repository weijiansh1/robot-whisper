# This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
#  See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
#  Author(s):       Vincent Rouvreau
#
# Copyright (C) 2016  Inria
#
# Modification(s):
#   - 2025/10 Vincent Rouvreau: Remove import_module, __available_modules and __mechanism_modules mechanism
#   - YYYY/MM Author: Description of the modification

__author__ = "GUDHI Editorial Board"
__copyright__ = "Copyright (C) 2016 Inria"
__license__ = "https://gudhi.inria.fr/licensing/"
__version__ = "3.13.0"
# This variable is used by doctest to find files
__root_source_dir__ = "/__w/gudhi-devel/gudhi-devel"
__debug_info__ =     "Python version 3.11.14\n" \
    "Nanobind version 2.13.0 \n" \
    "Numpy version 2.4.6 \n" \
    "Eigen3 version 3.3.4\n" \
    "Boost version 1.90.0\n" \
    "CGAL version 6.1.1\n" \
    "GMP_LIBRARIES = /usr/lib64/libgmp.so\n" \
    "GMPXX_LIBRARIES = /usr/lib64/libgmpxx.so\n" \
    "MPFR_LIBRARIES = /lib64/libmpfr.so\n" \


import os
from pathlib import Path

# Specific for Windows - required by datasets/generators that links with _default_cgal_random.dll in gudhi directory
if os.name == 'nt':
    os.add_dll_directory(Path(__file__).parent)

from .cubical_complex import CubicalComplex
from .periodic_cubical_complex import PeriodicCubicalComplex
from .simplex_tree import SimplexTree
from .rips_complex import RipsComplex
from .witness_complex import WitnessComplex
from .strong_witness_complex import StrongWitnessComplex
from .nerve_gic import CoverComplex

from .reader_utils import (
    read_lower_triangular_matrix_from_csv_file,
    read_persistence_intervals_grouped_by_dimension,
    read_persistence_intervals_in_dimension,
)

from .off_utils import (
    read_points_from_off_file,
    write_points_to_off_file,
)

from .persistence_graphical_tools import (
    plot_persistence_barcode,
    plot_persistence_diagram,
    plot_persistence_density
)

__all__ = [
    'CubicalComplex',
    'PeriodicCubicalComplex',
    'SimplexTree',
    'RipsComplex',
    'WitnessComplex',
    'StrongWitnessComplex',
    'CoverComplex',
    'read_lower_triangular_matrix_from_csv_file',
    'read_persistence_intervals_grouped_by_dimension',
    'read_persistence_intervals_in_dimension',
    'read_points_from_off_file',
    'write_points_to_off_file',
    'plot_persistence_barcode',
    'plot_persistence_diagram',
    'plot_persistence_density',
]

try:
    # if no CGAL
    from .bottleneck import bottleneck_distance

    __all__ += [
        'bottleneck_distance',
    ]
except ImportError:
    pass

try:
    # if no CGAL or no Eigen
    from .delaunay_complex import DelaunayComplex, DelaunayCechComplex, AlphaComplex

    from .euclidean_strong_witness_complex import EuclideanStrongWitnessComplex
    from .euclidean_witness_complex import EuclideanWitnessComplex

    from .tangential_complex import TangentialComplex
    
    from .subsampling import (
        choose_n_farthest_points,
        pick_n_random_points,
        sparsify_point_set,
        GUDHI_SUBSAMPLING_USE_CGAL,
    )

    __all__ += [
        'DelaunayComplex',
        'AlphaComplex',
        'DelaunayCechComplex',
        'EuclideanStrongWitnessComplex',
        'EuclideanWitnessComplex',
        'TangentialComplex',
        'choose_n_farthest_points',
        'pick_n_random_points',
        'sparsify_point_set',
    ]
except ImportError:
    pass


# Modules exposed at package level
from . import random

__all__ += [
    'random',
]
