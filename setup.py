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
    package_data={'rockfish': [] },
    long_description="""Really long.""",
)
