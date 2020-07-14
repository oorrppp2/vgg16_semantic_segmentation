from setuptools import setup
# from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(name='HoughVoting',
      ext_modules=[CppExtension('HoughVoting', ['houghvoting.cc'],
      # library_dirs = ['/home/user/anaconda3/lib/'])],
      library_dirs = ['/home/user/anaconda3/envs/PoseCNN/lib/'])],
      cmdclass={'build_ext': BuildExtension})

