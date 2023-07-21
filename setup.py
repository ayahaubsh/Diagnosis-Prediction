from setuptools import setup

REQUIREMENTS = [  
]

# DO NOT EDIT BELOW THIS LINE
DEV_REQUIREMENTS = [
    "pre-commit",
    "black",
    "flake8",
    "flake8-docstrings",
    "isort",
    "pep8-naming",
]

TEST_PACKAGES = [
    "pytest",
    "pytest-cov",
]

setup(
    name = "Diagnosis Prediction",
    version = "1.0.0",
    description = " Prediction of prognosis of patients based on their symptoms",
    author = ["Aya Haubsh"],
    author_email = ["ayahaubsh9@gmail.com"],
    python_requires = "==3.10.10",
    packages = ["src"],
    install_requires = REQUIREMENTS + DEV_REQUIREMENTS,
    extras_require = {
        "dev": DEV_REQUIREMENTS + TEST_PACKAGES,
        "test": TEST_PACKAGES,
    },
    include_package_data = True,
)
