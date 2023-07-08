from utils import open_nea_table, download_nea_table
from mass_year import mass_vs_year_plot, planet_discovery_stat
import pandas as pd
import numpy as np
import os
import sys

from datetime import date
import math

# %% [markdown]
# If needed to update the NEA table then first run  "download_nea_table"

# %%
nea_full_table = open_nea_table()
nea_full_table.head(3)

# %%
nea_full_table.columns.values

# %%
relevant_columns = ['pl_name', 'pl_letter', 'hostname', 'gaia_id', 'ra', 'dec', 'disc_year', 'discoverymethod',
       'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2', 
       'pl_orbsmax', 'pl_orbsmaxerr1','pl_orbsmaxerr2', 
       'pl_rade', 'pl_radeerr1', 'pl_radeerr2', 
       'pl_masse', 'pl_masseerr1', 'pl_masseerr2', 
       'pl_bmasse', 'pl_bmasseerr1', 'pl_bmasseerr2', 
       'pl_bmassprov', 
       'pl_msinie', 'pl_msinieerr1', 'pl_msinieerr2',
       'st_teff', 'st_met', 'st_lum', 'st_logg', 'st_age', 'st_mass', 'st_rad', 
       'ttv_flag', 'tran_flag', 'rv_flag']

# %%
nea_relevant = nea_full_table[relevant_columns]
nea_relevant.to_csv('nea_relevant.csv', index = False)

# %%
#nea_relevant[['pl_rade', 'pl_bmasse', 'disc_year', 'discoverymethod']].head(20)

# %% [markdown]
# R-M relation should be taken considering Teff, radius, and orbital period

# %%
tmp = nea_relevant[['pl_rade', 'st_teff', 'pl_bmasse', 'pl_orbper', 'st_rad']]
tmp = tmp[tmp.pl_bmasse == tmp.pl_bmasse]
tmp = tmp[tmp.pl_rade > 0]
tmp = tmp[tmp.st_teff > 0]
print (tmp)
tmp['st_rad'].isna().sum()

# %% [markdown]
# # Mass discovery year plot

# %%
planet_discovery_stat()

# %%
mass_vs_year_plot()

# %%
from PIL import Image

def change_resolution(input_path, output_path, scale):
    # Open the image
    image = Image.open(input_path)

    width, height = image.size
    print (width, height)

    new_width, new_height = int(width*scale), int(height*scale)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Save the resized image
    resized_image.save(output_path)

# Example usage
#change_resolution('./ss_planet_images/pngaaa.com-267808.png', 'nobel.png', 0.11)
#change_resolution('./ss_planet_images/pngaaa.com-107985.png', './ss_planet_images/earth.png', 0.1)
#change_resolution('./ss_planet_images/pngaaa.com-107985.png', './ss_planet_images/earth.png', 0.1)
#change_resolution('./ss_planet_images/pngaaa.com-107985.png', './ss_planet_images/earth.png', 0.1)


# %% [markdown]
# The images are taken (and then resized) from pngaa.com under License of "Non-comercial Use"

# %%



