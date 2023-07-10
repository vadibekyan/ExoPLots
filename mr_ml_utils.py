#!/usr/bin/python

import numpy as np
import pandas as pd

def bcv_from_teff(teff):
    """
    Calculate the Bolometric Correction in the V-band (BCV) based on the effective temperature (teff).
    Based on the work of Flower (1996) https://ui.adsabs.harvard.edu/abs/1996ApJ...469..355F/abstract 

    Parameters:
        teff (float or array-like): Effective temperature(s).

    Returns:
        float or array-like: Bolometric Correction(s) in the V-band (BCV).

    Raises:
        None.
    """
    # Convert teff to a NumPy array if it's not already
    teff = np.array(teff)
    
    # Initialize an array to store the BCV values
    bcv = np.zeros_like(teff)
    
    # Calculate BCV for teff < 5111
    condition = (teff < 5111)
    log_teff = np.log10(teff[condition])
    log_teff_squared = log_teff ** 2
    log_teff_cubed = log_teff ** 3
    bcv[condition] = -19053.7291496456 + 15514.4866764412 * log_teff - 4212.78819301717 * log_teff_squared + 381.476328422343 * log_teff_cubed
    
    # Calculate BCV for 5111 <= teff < 7943
    condition = (teff >= 5111) & (teff < 7943)
    log_teff = np.log10(teff[condition])
    log_teff_squared = log_teff ** 2
    log_teff_cubed = log_teff ** 3
    log_teff_quartic = log_teff_squared ** 2
    bcv[condition] = -37051.0203809015 + 38567.2629965804 * log_teff - 15065.1486316025 * log_teff_squared + 2617.24637119416 * log_teff_cubed - 170.623810323864 * log_teff_quartic
    
    # Calculate BCV for teff >= 7943
    condition = (teff >= 7943)
    log_teff = np.log10(teff[condition])
    log_teff_squared = log_teff ** 2
    log_teff_cubed = log_teff ** 3
    log_teff_quartic = log_teff_squared ** 2
    log_teff_quintic = log_teff_cubed ** 2
    bcv[condition] = -118115.450538963 + 137145.973583929 * log_teff - 63623.3812100225 * log_teff_squared + 14741.2923562646 * log_teff_cubed - 1705.87278406872 * log_teff_quartic + 78.873172180499 * log_teff_quintic
    
    # Return the calculated BCV values
    return bcv


def Vmag_to_L(teff, v, plx):
    """
    Convert the visual magnitude (Vmag) of a star to its luminosity (L).

    V(sun) = -26.76
    BCv(sun) = -0.08
    Mbol(sun) = 4.73

    Parameters:
        teff (float): Effective temperature of the star.
        v (float): Visual magnitude of the star.
        plx (float): Parallax of the star.

    Returns:
        float: Luminosity of the star.

    Raises:
        None.
    """

    # Calculate the Bolometric Correction in the V-band (BCV) using effective temperature
    bcv = bcv_from_teff(teff)

    # Calculate the absolute visual magnitude (Mv) using the visual magnitude (v) and parallax (plx)
    Mv = v + 5. + 5. * np.log10(plx / 1000.)

    # Calculate the bolometric magnitude (Mbol) by adding the BCV to Mv
    Mbol = Mv + bcv

    # Calculate the luminosity (L) using the bolometric magnitude (Mbol)
    L = 10 ** (-0.4 * (Mbol - 4.73))

    # Return the calculated luminosity (L)
    return L


def a_from_P(Mstar, P):
    """
    Calculates the semimajor axis (a) of the orbit using stellar mass and orbital period
    using Kepler's Third Law.
    Mass of the planet is ignored, as this will be used for planets without mass determination.
    ***Reminder*** all these are JUST to make some plots.

    Parameters:
        Mstar: Stellar mass in solar masses (M☉).
        P: Orbital period in days.

    Returns:
        a: Semimajor axis of the orbit in astronomical units (AU).
    """
    P_years = P / 365.25  # Convert orbital period from days to years.
    G = 39.478  # Gravitational constant in AU^3 M☉^-1 yr^-2 units.
    a = (P_years ** 3 * G * Mstar / (4 * np.pi ** 2)) ** (1 / 3)

    return a



def Teq_from_teff_v_plx_a(teff, v, plx, a):
    """
    Calculate the APPROXIMATE equilibrium temperature (Teq) of a planet based on the effective temperature (teff) of its host star,
    the visual magnitude (v) of the star, the parallax (plx), and the semimajor axis (a) of the planet's orbit.
    This assumes zero Bond albedo. See [Wang & Dai (2017](https://ui.adsabs.harvard.edu/abs/2018ApJ...860..175W/abstract)

    Parameters:
        teff (float): Effective temperature of the host star.
        v (float): Visual magnitude of the star.
        plx (float): Parallax of the star.
        a (float): Semimajor axis of the planet's orbit in AU.

    Returns:
        float: Equilibrium temperature (Teq) of the planet.
    """

    # Calculate the luminosity (L) of the star using the Vmag_to_L function
    L = Vmag_to_L(teff, v, plx)

    # Calculate the incident flux (ins_flux) received by the planet
    ins_flux = L / (a ** 2)

    # Calculate the equilibrium temperature (Teq) of the planet
    Teq = (ins_flux ** 0.25) * 280

    # Return the calculated equilibrium temperature (Teq)
    return Teq



def Teq_from_teff_v_plx_P(teff, v, plx, P, Mstar):
    """
    Calculate the APPROXIMATE equilibrium temperature (Teq) of a planet based on the effective temperature (teff) of its host star,
    the visual magnitude (v) of the star, the parallax (plx), and the semimajor axis (a) of the planet's orbit.
    This assumes zero Bond albedo. See [Wang & Dai (2017](https://ui.adsabs.harvard.edu/abs/2018ApJ...860..175W/abstract)

    Parameters:
        teff (float): Effective temperature of the host star.
        v (float): Visual magnitude of the star.
        plx (float): Parallax of the star.
        P (float): Orbital period of the planet in days

    Returns:
        float: Equilibrium temperature (Teq) of the planet.
    """

    # Calculate the luminosity (L) of the star using the Vmag_to_L function
    L = Vmag_to_L(teff, v, plx)

    # Calculate semimajor axis from period
    a = a_from_P(Mstar, P)

    # Calculate the incident flux (ins_flux) received by the planet
    ins_flux = L / (a ** 2)

    # Calculate the equilibrium temperature (Teq) of the planet
    Teq = (ins_flux ** 0.25) * 280

    # Return the calculated equilibrium temperature (Teq)
    return Teq


def Teq_from_L_a(L, a):
    """
    Calculate the APPROXIMATE equilibrium temperature (Teq) of a planet based on the Luminosity (L) of its host star
    and the semimajor axis (a) of the planet's orbit.
    This assumes zero Bond albedo. See [Wang & Dai (2017](https://ui.adsabs.harvard.edu/abs/2018ApJ...860..175W/abstract)

    Parameters:
        L: (float): Luminosity of the host star in soloar units
        a (float): Semimajor axis of the planet's orbit in AU.

    Returns:
        float: Equilibrium temperature (Teq) of the planet.
    """

    # Calculate the incident flux (ins_flux) received by the planet
    ins_flux = L / (a ** 2)

    # Calculate the equilibrium temperature (Teq) of the planet
    Teq = (ins_flux ** 0.25) * 280

    # Return the calculated equilibrium temperature (Teq)
    return Teq


def teq_hellper_function(x, pl_orbsmax, pl_orbper, st_teff, sy_vmag, sy_plx, st_mass, Teq):
    """
    Helper function to calculate the equilibrium temperature (Teq) based on different parameters.

    Parameters:
        x (array-like): Input array or DataFrame row containing the values of the parameters.
        pl_orbsmax (str): Column name or key for the planet's semimajor axis parameter.
        pl_orbper (str): Column name or key for the planet's orbital period parameter.
        st_teff (str): Column name or key for the stellar effective temperature parameter.
        sy_vmag (str): Column name or key for the system's visual magnitude parameter.
        sy_plx (str): Column name or key for the system's parallax parameter.
        st_mass (str): Column name or key for the stellar mass parameter.
        Teq (str): Column name or key for the equilibrium temperature (output) parameter.

    Returns:
        x (array-like): Input array or DataFrame row with the equilibrium temperature (Teq) added as a new column.

    Note:
        The function uses the `Teq_from_teff_v_plx_a` and `Teq_from_teff_v_plx_P` functions to calculate Teq.
        The resulting Teq value is rounded to the nearest integer using `np.round`.
    """
    if x[pl_orbsmax] > 0:
        # Calculate Teq using Teq_from_teff_v_plx_a function
        x[Teq] = Teq_from_teff_v_plx_a(x[st_teff], x[sy_vmag], x[sy_plx], x[pl_orbsmax])
    else:
        # Calculate Teq using Teq_from_teff_v_plx_P function
        x[Teq] = Teq_from_teff_v_plx_P(x[st_teff], x[sy_vmag], x[sy_plx], x[pl_orbper], x[st_mass])
    
    # Round the Teq value to the nearest integer
    x[Teq] = np.round(x[Teq])
    
    return x


def creating_MR_ML_table(nea_full_table):
    """
    Create a table with relevant columns for mass-radius relation analysis and calculate the equilibrium temperature (Teq).

    Parameters:
        nea_full_table (DataFrame): Full table containing the data.

    Returns:
        nea_MR_final_table (DataFrame): Final table with relevant columns: 'pl_rade', 'pl_bmasse', 'Teq'.
    """
    # Select relevant columns from the full table
    relevant_columns = ['pl_rade', 'pl_orbsmax', 'pl_bmasse', 'pl_orbper', 'sy_vmag', 'sy_plx', 'st_teff', 'st_mass']
    nea_relevant = nea_full_table[relevant_columns]

    # Filter rows with positive values for mass and radius
    nea_with_M_and_R = nea_relevant[(nea_relevant.pl_bmasse > 0) & (nea_relevant.pl_rade > 0)].reset_index(drop=True)

    # Remove stars with negative parallax
    nea_with_M_and_R = nea_with_M_and_R[nea_with_M_and_R.sy_plx > 0]

    # Remove rows with NaN values except for 'pl_orbsmax' column
    nea_with_M_and_R_cleaned = nea_with_M_and_R.dropna(subset=nea_with_M_and_R.columns.drop('pl_orbsmax'))

    # Add 'Teq' column and initialize with 0
    nea_with_M_and_R_cleaned['Teq'] = 0

    # Calculate the equilibrium temperature 'Teq' using a helper function
    nea_with_M_and_R_cleaned = nea_with_M_and_R_cleaned.apply(lambda x: teq_hellper_function(x, 'pl_orbsmax', 'pl_orbper', 'st_teff', 'sy_vmag', 'sy_plx', 'st_mass', 'Teq'), axis=1)

    # Select the final columns for the output table and reset the index
    nea_MR_final_table = nea_with_M_and_R_cleaned[['pl_rade', 'pl_bmasse', 'Teq']].reset_index(drop=True)

    return nea_MR_final_table




