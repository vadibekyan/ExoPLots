from utils import open_nea_table, download_nea_table
import pandas as pd
import numpy as np
import os
from datetime import date
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def creating_data_to_plot():
    # Function to create data for plotting

    # Call a function to open the NEA table and retrieve the full table
    nea_full_table = open_nea_table()

    # Select relevant columns from the NEA table
    relevant_columns = ['pl_orbper', 'discoverymethod', 'pl_bmasse']
    nea_relevant = nea_full_table[relevant_columns]

    # Select planets with masses greater than 0
    nea_mass_period_sample = nea_relevant[nea_relevant.pl_bmasse > 0].reset_index()

    # Planets discovered by Transit method or TTV are considered as Transiting planets
    nea_mass_period_Transit = nea_mass_period_sample[(nea_mass_period_sample.discoverymethod == 'Transit') | (nea_mass_period_sample.discoverymethod == 'Transit Timing Variations')]
    nea_mass_period_RV = nea_mass_period_sample[nea_mass_period_sample.discoverymethod == 'Radial Velocity']
    nea_mass_period_Imaging = nea_mass_period_sample[nea_mass_period_sample.discoverymethod == 'Imaging']
    nea_mass_period_Microlensing = nea_mass_period_sample[nea_mass_period_sample.discoverymethod == 'Microlensing']

    # Selecting planets detected by "Other" methods
    rv_trainsiting_imaging_microlensing = pd.concat([nea_mass_period_RV, nea_mass_period_Transit, nea_mass_period_Imaging, nea_mass_period_Microlensing])
    nea_mass_period_Other = nea_mass_period_sample.drop(rv_trainsiting_imaging_microlensing.index)

    # Return the created data for plotting
    return nea_mass_period_Other, nea_mass_period_Transit, nea_mass_period_RV, nea_mass_period_Microlensing, nea_mass_period_Imaging, nea_mass_period_sample, nea_full_table



def imscatter(x, y, image, ax=None, zoom=1):
    """
    Scatter images on a plot.

    Parameters:
    - x: The x-coordinates of the images.
    - y: The y-coordinates of the images.
    - image: The image file or array to scatter.
    - ax (optional): The axes on which to scatter the images. If not provided, the current axes will be used.
    - zoom (optional): The zoom level of the images.

    Returns:
    - artists: A list of artists representing the scattered images.

    P.S. This is a modified version of a function taken from internet (do not remember from where)
    """

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

    # Iterate over x and y values
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    # Update data limits based on x and y values
    ax.update_datalim(np.column_stack([x, y]))

    # Return the artists
    return artists


# Call the function and assign the returned values to variables
nea_mass_period_Other, nea_mass_period_Transit, nea_mass_period_RV, nea_mass_period_Microlensing, nea_mass_period_Imaging, nea_mass_period_sample, nea_full_table = creating_data_to_plot()


def creating_data_for_GMM(nea_mass_period_sample):

    mass_period = nea_mass_period_sample[['pl_orbper', 'pl_bmasse']]
    mass_period = mass_period[(mass_period.pl_orbper > 0)]

    mass_period['log_mass'] = np.log10(mass_period.pl_bmasse)
    mass_period['log_period'] = np.log10(mass_period.pl_orbper)
    mass_period_log = mass_period[['log_period', 'log_mass']]

    return mass_period_log


def mass_period_plot(SS_planets = True, GMM = True):
    """
    Generate a scatter plot of planet mass versus year of discovery.

    Returns:
    - None (saves image)
    """

    sns.set_style('ticks')
    sns.set_style("white")
    plt.rcParams['legend.handlelength'] = 0.1


    # Create the figure and adjust its settings
    fig = plt.figure(1, figsize=(9, 6), dpi=200)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.99)

    # Set up the subplot
    ax = fig.add_subplot(1, 1, 1)

    # Scatter plots for different methods
    ax.scatter(nea_mass_period_Other.pl_orbper, nea_mass_period_Other.pl_bmasse, color='green', s=70, alpha=0.99, label='Other')
    ax.scatter(nea_mass_period_Transit.pl_orbper, nea_mass_period_Transit.pl_bmasse, color='royalblue', s=70, alpha=0.9, label='Transit/TTV')
    ax.scatter(nea_mass_period_RV.pl_orbper, nea_mass_period_RV.pl_bmasse, color='red', s=70, alpha=0.4, label='RV')
    ax.scatter(nea_mass_period_Microlensing.pl_orbper, nea_mass_period_Microlensing.pl_bmasse, color='yellow', s=70, alpha=0.4, label='Microlensing')
    ax.scatter(nea_mass_period_Imaging.pl_orbper, nea_mass_period_Imaging.pl_bmasse, color='purple', s=70, alpha=0.99, label='Imaging')


    # Set tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2)

    # Set y-axis scale to logarithmic
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Set y-axis and x-axis limits
    ax.set_xlim(10**-1.5, 10**9.)
    ax.set_ylim(10**-1.4, 10**4.6)


    # Set tick parameters for the main subplot
    ax.tick_params(axis='both', top='on', bottom='on', right='on', left='on', labelsize=18)

    # Set legend properties
    ax.legend(loc='best', fontsize=15, labelspacing=0.1)


    # Set x and y labels
    ax.set_ylabel('Mass ($M_{\mathrm{\oplus}}$)', fontsize=20)
    ax.set_xlabel('Period (days)', fontsize=20)


    if SS_planets == True:

        # Set image paths and scatter images
        current_dir = os.getcwd()

        x = 88
        y = 0.055
        image_path = os.path.join(current_dir, 'ss_planet_images', 'mercury.png')
        imscatter(x, y, image_path, zoom=0.15, ax=ax)

        x = 224
        y = 0.815
        image_path = os.path.join(current_dir, 'ss_planet_images','venus.png')
        imscatter(x, y, image_path, zoom=0.2, ax=ax)

        x = 365
        y = 1.0
        image_path = os.path.join(current_dir, 'ss_planet_images', 'earth.png')
        imscatter(x, y, image_path, zoom=0.4, ax=ax)

        x = 687
        y = 0.107
        image_path = os.path.join(current_dir, 'ss_planet_images','mars.png')
        imscatter(x, y, image_path, zoom=0.4, ax=ax)

        x = 4332
        y = 317
        image_path = os.path.join(current_dir, 'ss_planet_images', 'jupiter.png')
        imscatter(x, y, image_path, zoom=0.7, ax=ax)

        x = 10759
        y = 95
        image_path = os.path.join(current_dir, 'ss_planet_images', 'saturn.png')
        imscatter(x, y, image_path, zoom=0.4, ax=ax)

        x = 31000
        y = 14.6
        image_path = os.path.join(current_dir, 'ss_planet_images','uranus.png')
        imscatter(x, y, image_path, zoom=0.25, ax=ax)

        x = 60152
        y = 17.2
        image_path = os.path.join(current_dir, 'ss_planet_images', 'neptune.png')
        imscatter(x, y, image_path, zoom=0.7, ax=ax)
   

    if GMM == True:
        
        from sklearn.mixture import GaussianMixture
        from matplotlib.patches import Ellipse
        import matplotlib.patches as mpatches

        # Scatter plot for the secondary subplot (ax2)
        ax2 = fig.add_subplot(1, 1, 1, frame_on=False)

        # Sample 2D data (Replace this with your actual data)
        mass_period_log = creating_data_for_GMM(nea_mass_period_sample)
        data = mass_period_log.values

        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(data)

        # Get cluster assignments and cluster centroids
        #cluster_labels = gmm.predict(data)
        cluster_centroids = gmm.means_

        # Create a scatter plot for each cluster
        colors = ['r', 'g', 'b']

        # Plot the cluster centroids
        ax2.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], marker='X', s=50, c='k')


        for i in range(3):
            cov_matrix = gmm.covariances_[i]
            mean_vector =  gmm.means_[i]

            # Calculate ellipse properties
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width = 2.5 * np.sqrt(2 * eigenvalues[0])
            height = 2.5 * np.sqrt(2 * eigenvalues[1])

            # Create ellipse
            ellipse = Ellipse(mean_vector, width, height, angle = angle, fill=False, edgecolor='k', lw=3, alpha=1)
            ax2.add_patch(ellipse)

        # Set subplot properties
        ax2.set_yscale('linear')
        ax2.set_xscale('linear')
        ax2.set_xlim(-1.5, 9.)
        ax2.set_ylim(-1.4, 4.6)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        # Ellipse as a legend
        ellipse_legend_centroid_x = 7.8
        ellipse_legend_centroid_y = 0.5

        # Create the ellipse centered at the origin
        ellipse = mpatches.Ellipse((ellipse_legend_centroid_x, ellipse_legend_centroid_y), 2.2, 0.5, fill=False, edgecolor='k', lw=2)
        ax2.scatter(ellipse_legend_centroid_x-0.8, ellipse_legend_centroid_y, marker='X', s=50, c='k')
        # Add the ellipse and cross
        ax2.add_patch(ellipse)
        ax2.text(ellipse_legend_centroid_x+0.1, ellipse_legend_centroid_y, 'GMM Clusters', fontsize=12, ha='center', va='center', color='black')




    # Save the plot as an image
    plt.savefig('mass_period.png', facecolor='White')



if __name__ == '__main__':
    mass_period_plot(SS_planets = True, GMM = True)