from distutils.core import setup

setup(
    name='swordfish',
    version='0.1',
    description='Your sword for Fisher forecasting.',
    author='Thomas Edwards and Christoph Weniger',
    author_email='c.weniger@uva.nl',
    packages=['swordfish'],
    package_data={'swordfish': [] },
    long_description="""swordfish is a Python tool to study the information gain of counting experiments.""",
)
