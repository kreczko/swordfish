from distutils.core import setup #, Extension
#import numpy
#from Cython.Distutils import build_ext
#import os

#os.environ["CC"] = "g++"
#os.environ["CXX"] = "g++"

setup(
    name='swordfish',
    version='0.1',
    description='Your sword for Fisher forecasting that rocks',
    author='Thomas Edwards and Christoph Weniger',
    author_mail='c.weniger@uva.nl',
    packages=['swordfish'],
    package_data={'swordfish': [] },
    long_description="""Really long.""",
)
