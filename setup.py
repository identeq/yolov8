"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2023-01-24
"""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []
try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except IOError as e:
    print(e)
setup(
    name='yolov8',
    version='1.0',
    description="",
    long_description=readme + '\n',
    author="Pravesh Kaji Budhathoki",
    author_email='pravesh.buddhathoki@treeleaf.ai',
    url='',
    packages=find_packages(),
    package_dir={},
    package_data={'': ['*.yaml']},  # include all .yaml file from ultralytics dir
    install_requires=requirements,
    license="",
    zip_safe=False,
    keywords=''
)
