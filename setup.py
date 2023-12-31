from setuptools import find_packages, setup


HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> list[str]:
    with open(file_path) as file_obj:
        # Use a list comprehension to strip newlines and exclude unwanted items in one step
        requirements = [req.strip() for req in file_obj if req.strip() != HYPHEN_E_DOT]

    return requirements


setup(
    name="fpl_project",
    version="0.0.1",
    author="Beechhceeb",
    author_email="beechhceeb@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
