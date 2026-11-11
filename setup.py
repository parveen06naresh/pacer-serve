"""Builds the optional C++17 extension pacer._core. If no compiler is available the
pure-Python scheduler is used instead (pacer.core.HAVE_CORE is False)."""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[Pybind11Extension("pacer._core", ["pacer/csrc/core.cpp"], cxx_std=17,
                                   extra_compile_args=["-O3"])],
    cmdclass={"build_ext": build_ext},
)
