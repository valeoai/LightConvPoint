from setuptools import setup
from torch.utils import cpp_extension
import sys

print(sys.argv)

if "--compile" in sys.argv:
    print("LIGHTCONVPOINT -- COMPILING CPP MODULES")
    ext_modules=[
       cpp_extension.CppExtension(
           "lightconvpoint.knn_c_func",
           [
               "lightconvpoint/src/knn.cxx",
               "lightconvpoint/src/knn_bind.cxx",
               "lightconvpoint/src/knn_random.cxx",
               "lightconvpoint/src/knn_farthest.cxx",
               "lightconvpoint/src/knn_convpoint.cxx",
               "lightconvpoint/src/knn_quantized.cxx",
           ],
           extra_compile_args=["-fopenmp"],
           extra_link_args=["-fopenmp"],
       )
    ]
    cmdclass={"build_ext": cpp_extension.BuildExtension}
    sys.argv.remove("--compile")
else:
    print("LIGHTCONVPOINT -- PYTHON MODULES")
    ext_modules=[]
    cmdclass={}

setup(
    name="lightconvpoint",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
