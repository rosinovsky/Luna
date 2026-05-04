from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "clinker_forecast",
        [
            "bindings/python/clinker_forecast.cpp",
            "core/src/model.cpp",
            "core/src/inference.cpp",
            "core/src/export.cpp",
            "core/src/physics.cpp",
            "core/src/ops/ops.cpp",
        ],
        include_dirs=[
            "core/include",
            pybind11.get_include(),
        ],
        cxx_std=17,
        # Оптимизации
        extra_compile_args=["-O3", "-ffast-math", "-funroll-loops"],
    ),
]

setup(
    name="clinker-forecast",
    version="1.0.0",
    author="ClinkerForecast Team",
    description="Neural forecasting engine for cement clinker quality",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)