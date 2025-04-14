from setuptools import setup, find_packages
from typing import List



def get_requirements(file_path:str)-> list[str]:
    """
    This function returns a list of requirements from the given file path.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove any leading/trailing whitespace characters
        requirements = [req.strip() for req in requirements]
        # Remove any version specifiers (e.g., 'package==1.0.0')
        requirements = [req.split('==')[0] for req in requirements if req and not req.startswith('#')]

        if "-e ." in requirements:
            requirements.remove("-e .")
    
    return requirements



# This is the setup script for the mlproject package.
setup(
    name='mlproject',
    version='0.1',
    author='nimnusmik',
    author_email='kimsunmin0227@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)