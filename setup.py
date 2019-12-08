import os
import codecs

from setuptools import setup, find_packages

def readme():
    with codecs.open('README.rst', encoding='utf-8-sig') as f:
        return f.read()

version_file= os.path.join('ehgfeatures', '_version.py')
with open(version_file) as f:
    exec(f.read())

DISTNAME= 'ehgfeatures'
DESCRIPTION= 'EHG features'
LONG_DESCRIPTION= readme()
LONG_DESCRIPTION_CONTENT_TYPE='text/x-rst'
MAINTAINER= 'Gilles Vandewiele and Gyorgy Kovacs'
MAINTAINER_EMAIL= 'gilles.vandewiele@ugent.be; gyuriofkovacs@gmail.com'
URL= 'https://github.com/GillesVandewiele/EHG-Oversampling/'
LICENSE= 'MIT'
DOWNLOAD_URL= 'https://github.com/GillesVandewiele/EHG-Oversampling/'
VERSION= __version__
CLASSIFIERS= [  'Intended Audience :: Science/Research',
                'Intended Audience :: Developers',
                'Development Status :: 3 - Alpha',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Topic :: Software Development',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX',
                'Operating System :: Unix',
                'Operating System :: MacOS']
INSTALL_REQUIRES= ['wfdb', 'numpy', 'pandas', 'scipy', 'PyEMD', 'PyWavelets', 'neurokit', 'entropy']
EXTRAS_REQUIRE= {'tests': ['nose'],
                 'docs': ['sphinx', 'sphinx-gallery', 'sphinx_rtd_theme', 'matplotlib', 'pandas']}
PYTHON_REQUIRES= '>=3.5'
TEST_SUITE='nose.collector'
TEST_REQUIRES=['']
PACKAGE_DIR= {'ehgfeatures': 'ehgfeatures'}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      author=MAINTAINER,
      author_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      python_requires=PYTHON_REQUIRES,
      test_suite=TEST_SUITE,
      package_dir=PACKAGE_DIR,
      packages=find_packages(exclude=[]),
      test_requires=TEST_REQUIRES)
