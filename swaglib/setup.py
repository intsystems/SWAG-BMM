from setuptools import setup, find_packages

setup(
    name='swaglib', 
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        
    ],
    author='Andron00e, BungaBonga, lxstsvnd, alshestt',  
    author_email='semenov.andrei.v@gmail.com',
    description='a library for the evidence lower bound approach implementation for the Bayesian inference over deep neural networks',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache 2.0 License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',

)