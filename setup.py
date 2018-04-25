from distutils.core import setup

setup(name='bsbp',
      packages=['bsbp'],
      version='0.1',
      description='Implementation of the best subset maximum score binary prediction method proposed by Chen and Lee (2017).',
      url='https://github.com/LeyuChen/Best-Subset-Binary-Prediction',
      download_url='https://github.com/LeyuChen/Best-Subset-Binary-Prediction',
      author='Thu Pham',
      author_email='thp2107@gmail.com',
      package_data={'bsbp': ['*.csv']},
      include_package_data=True,
      classifiers = ['Programming Language :: Python :: 3.6'],
      python_requires='==3.6.*',
      zip_safe=False)
