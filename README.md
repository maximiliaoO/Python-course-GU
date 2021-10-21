# Python-course-GU

The Python script in this repository can be used to:
1. detect water-oil droplets or water-oil-water droplets (referred to as double-emulsion droplets) in a microscope picture
2. characterise the detected droplets (e.g. calculate volume of droplet)
3. summarise droplet statistics 

The script will create a histgramm and several CSV files containing information about the droplets

The repository contains (besides this README file):
- a license file
- the python script DE_statistics.py
- a microscopy image DEimage.tif
- an example CSV file df_droplets.csv (ideally this CSV file should be created by image recognition)

The script requires the following modules:
- import cv2
- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt

How to run:
- download files "DE_statistics.py", "DEimage.tif", "df_droplets.csv"
- run in terminal with "python DE_statistics.py"
