from setuptools import setup
from torch.utils import cpp_extension
import sys

print(sys.argv)

if "--nocompile" in sys.argv:
    print("LIGHTCONVPOINT -- PYTHON MODULES")
    ext_modules=[]
    cmdclass={}
    sys.argv.remove("--nocompile")
else:
    print("LIGHTCONVPOINT -- COMPILING CPP MODULES")
    ext_modules=[
       cpp_extension.CppExtension(
           "lightconvpoint.knn_c_func",
           [
               "lightconvpoint/src/functions_bind.cxx",
               "lightconvpoint/src/knn.cxx",
               "lightconvpoint/src/sampling_random.cxx",
               "lightconvpoint/src/sampling_farthest.cxx",
               "lightconvpoint/src/sampling_convpoint.cxx",
               "lightconvpoint/src/sampling_quantized.cxx",
           ],
           extra_compile_args=["-fopenmp"],
           extra_link_args=["-fopenmp"],
       )
    ]
    cmdclass={"build_ext": cpp_extension.BuildExtension}

setup(
    name="lightconvpoint",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
