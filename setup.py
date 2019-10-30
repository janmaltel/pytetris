from setuptools import setup

setup(name='pytetris',
      version='0.1',
      description='Tetris implementation tailored for use in reinforcement learning applications',
      # url='http://github.com/storborg/funniest',
      author='Jan Malte Lichtenberg',
      author_email='j.m.lichtenberg@bath.ac.uk',
      license='MIT',
      packages=['tetris'],
      zip_safe=False,
      install_requires=[
            'numpy',
            'numba'
      ])
