from setuptools import setup, find_packages

REQUIRED = ['numpy',
            'pandas',
            'matplotlib',
            'stable-baselines',
            'gym<0.20',
            'tensorflow<2.0',
            'tqdm',
            'cpython<3.0']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='mbcd',
    version='0.1',
    packages=['mbcd',],
    install_requires=REQUIRED,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    long_description=long_description,
    url='https://github.com/LucasAlegre/mbcd',
    license="MIT",
    description='Model-Based Reinforcement Learning Context Detection.'
)
