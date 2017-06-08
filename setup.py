from distutils.core import setup #, Extension
#import numpy
#from Cython.Distutils import build_ext
#import os

#os.environ["CC"] = "g++"
#os.environ["CXX"] = "g++"

setup(
    name='rockfish',
    version='0.1',
    description='Fisher forecasting that rocks',
    author='Thomas Edwards and Christoph Weniger',
    author_mail='c.weniger@uva.nl',
    packages=['rockfish'],
    #package_data={'rockfish': ['redshift_table.dat'] },
    long_description="""Really long.""",
#    cmdclass={'build_ext': build_ext},
#    ext_modules=[Extension("vader.helpers",
#                 sources=["vader/helpers.pyx", "src/los_integral.cpp"],
#                 include_dirs=[numpy.get_include(), 'src/'], language="c++")],
)

#setup(
#    name='yoda',
#    version='0.1',
#    description='Yield Optimization for the Dark matter Anarchist',
#    author='Christoph Weniger and Tom Edwards',
#    author_mail='c.weniger@uva.nl',
#    packages=['yoda'],
#    package_data={'yoda': [] },
#    long_description="""Really long.""",
#    cmdclass={'build_ext': build_ext},
#   ext_modules=[Extension("vader.helpers",
#                sources=["vader/helpers.pyx", "src/los_integral.cpp"],
#                include_dirs=[numpy.get_include(), 'src/'], language="c++")],
#)
