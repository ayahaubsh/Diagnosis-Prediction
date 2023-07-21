#!/usr/bin/env bash

conda env create -f environment.yaml
conda activate Diagnosis-Prediction
python -m pip install --upgrade pip
pip install --upgrade pip setuptools wheel
python -m pip install -e .
pre-commit install
