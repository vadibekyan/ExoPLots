from utils import open_nea_table, download_nea_table
import pandas as pd
import numpy as np
import os
from datetime import date
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



nea_full_table = open_nea_table()

relevant_columns = ['disc_year', 'discoverymethod', 'pl_bmasse']
nea_relevant = nea_full_table[relevant_columns]


# select planets with masses
nea_mass_year_sample =  nea_relevant[nea_relevant.pl_bmasse > 0].reset_index()


#Planets discovered by Transit method or TTV are considered as Transiting planets
nea_mass_year_Transit = nea_mass_year_sample[(nea_mass_year_sample.discoverymethod == 'Transit') | (nea_mass_year_sample.discoverymethod == 'Transit Timing Variations')]
nea_mass_year_RV = nea_mass_year_sample[nea_mass_year_sample.discoverymethod == 'Radial Velocity']
nea_mass_year_Imaging = nea_mass_year_sample[nea_mass_year_sample.discoverymethod == 'Imaging']
nea_mass_year_Microlensing = nea_mass_year_sample[nea_mass_year_sample.discoverymethod == 'Microlensing']


# Selecting planets detected by "Other" methods
rv_trainsiting_imaging_microlensing = pd.concat([nea_mass_year_RV, nea_mass_year_Transit, nea_mass_year_Imaging, nea_mass_year_Microlensing])
nea_mass_year_Other = nea_mass_year_sample.drop(rv_trainsiting_imaging_microlensing.index)


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    return artists

def mass_vs_year_plot():
    sns.set_style('ticks')
    sns.set_style("white")
    plt.rcParams['legend.handlelength'] = 0.1


    # The lower limit of x is fixed to 1988
    # the upper limit will be set one year after the current date
    today = date.today()
    current_year = today.year


    fig = plt.figure(1, figsize = (9,6), dpi = 200)
    ax = plt.gca()
    fig.subplots_adjust(left=0.1, right=0.98, bottom = 0.12, top = 0.99)

    ax = plt.subplot(1, 1, 1)

    ax.scatter(nea_mass_year_Other.disc_year, nea_mass_year_Other.pl_bmasse,  color = 'green', s=70, alpha = 0.99, label='Other')
    ax.scatter(nea_mass_year_Transit.disc_year, nea_mass_year_Transit.pl_bmasse,  color = 'royalblue', s=70, alpha = 0.9, label='Transit/TTV')
    ax.scatter(nea_mass_year_RV.disc_year, nea_mass_year_RV.pl_bmasse,  color = 'red', s=70, alpha = 0.4, label='RV')
    ax.scatter(nea_mass_year_Microlensing.disc_year, nea_mass_year_Microlensing.pl_bmasse,  color = 'yellow', s=70, alpha = 0.4, label='Microlensing')
    ax.scatter(nea_mass_year_Imaging.disc_year, nea_mass_year_Imaging.pl_bmasse,  color = 'black', s=70, alpha = 0.99, label='Imaging')

    plt.hlines(y=1, xmin=1991, xmax=current_year+1, linewidth=1, color = 'grey', linestyle='--')
    plt.hlines(y=17.15, xmin=1991, xmax=current_year+1, linewidth=1, color = 'grey', linestyle='--')
    plt.hlines(y=317, xmin=1991, xmax=current_year+1, linewidth=1, color = 'grey', linestyle='--')
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2)


    current_dir = os.getcwd()
    images_directory = os.path.join(current_dir, 'ss_planet_images')

    x = 1992.4
    y = 1.0
    image_path = os.path.join(current_dir, 'ss_planet_images', 'earth.png')
    imscatter(x, y, image_path, zoom=0.4, ax=ax)

    x = 1992.4
    y = 317
    image_path = os.path.join(current_dir, 'ss_planet_images', 'jupiter.png')
    imscatter(x, y, image_path, zoom=0.7, ax=ax)

    x = 1992.4
    y = 17.15
    image_path = os.path.join(current_dir, 'ss_planet_images', 'neptune.png')
    imscatter(x, y, image_path, zoom=0.7, ax=ax)

    x = 1995
    y = 150
    image_path = os.path.join(current_dir, 'ss_planet_images', 'nobel.png')
    imscatter(x, y, image_path, zoom = 0.8, ax=ax)

    ax.set_yscale('log')
    ymin = np.min(np.log10(nea_mass_year_sample.pl_bmasse)) - 0.2
    ymax = np.max(np.log10(nea_mass_year_sample.pl_bmasse)) + 0.2

    plt.ylim(10**ymin,10**ymax)
    plt.xlim(1991,current_year+1)


    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(loc=(0.25, 0.04), fontsize = 16, labelspacing=0.1)

    plt.tick_params(axis='both', 
    top='on', bottom = 'on', right = 'on', left = 'on')

    plt.ylabel('Mass ($M_{\mathrm{\oplus}}$)', fontsize = 20)
    plt.xlabel('Year of discovery', fontsize = 20)

    plt.savefig('mass_vs_year.png', facecolor = 'White')


if __name__ == '__main__':
    mass_vs_year_plot()