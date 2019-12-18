from setuptools import setup

setup(
    name='avocado',
    version='0.1.0',
    packages=['avocado'],
    include_package_data=True,
    install_requires=[
        'numpy ==1.16.5',
        'matplotlib==3.1.1',
        'scikit-learn==0.21.3',
        'scipy==1.3.1',
        'pandas==0.25.1',
        'click==7.0',
        'pycodestyle==2.5.0',
        'pydocstyle==4.0.1',
        'pylint==2.4.3',
        'pytest==5.2.1',
    ],
)
