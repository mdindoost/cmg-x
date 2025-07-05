from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy as np
import os

cmg_module = Extension(
    name="cmgx._cmgcore",  # compiled shared object goes in cmgx/
    sources=["src/cmgCluster.c"],
    include_dirs=["include", np.get_include()],
    extra_compile_args=["-O3", "-fPIC"],
)

setup(
    name="cmgx",
    version="0.1.0",
    author="Mohammad Dindoost",
    author_email="your.email@example.com",  # (optional)
    description="Multiscale Graph Coarsening via Combinatorial Multigrid",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    license="MIT",  # Or Apache-2.0
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=["numpy", "scipy", "torch", "torch-geometric"],
    ext_modules=[cmg_module],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=False,
)

