from setuptools import setup, find_packages

setup(
    name='FRAP_Analysis',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/DomenicN/FRAP_Analysis',
    license='MIT',
    author='Domenic Narducci',
    author_email='domenicn@mit.edu',
    description='A simple package for FRAP analysis.',
	entry_points = {
		'console_scripts': [
			'frap=FRAP_Analysis.__main__:cli'
		],
	}
)
