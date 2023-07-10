from utils import open_nea_table, download_nea_table
from mass_year import mass_vs_year_plot, planet_discovery_stat
import pandas as pd
import numpy as np
import os
import sys
from datetime import date
import math

from mr_ml_utils import creating_MR_ML_table


#If needed to update the NEA table then first run  "download_nea_table"
nea_full_table = open_nea_table()

# Creating the final table with R, M, and Teq.
MR_ML_table = creating_MR_ML_table(nea_full_table)

print (MR_ML_table)