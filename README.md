# ExoPLots

Several codes (**mass_year**, **mass_period**) to visualize exoplanet data extracted from the NASA Exoplanet Archive



## Mass vs Year

**mass_year** has two potentially useful functions.

``planet_discovery_stat()``  displays the number of discovered planets by different methods:


```python
planet_discovery_stat()

-----  Some Stats -----
5463 planets discovered so far

1088 via Transiting/TTV methods
1043 via RV method
200 via Microlensing method
64 via Imaging method
35 by other methods
-----     End     ----
```
``mass_vs_year_plot()``  plots the mass of the discovered planets as a function of the year of their discovery.

<img src="https://github.com/vadibekyan/ExoPLots/assets/25388659/c878719a-9464-489f-a22f-ed83dfd28dcd" alt="Figure" width="600" height="400">

``mass_period_plot()``  plots the mass of the discovered planets as a function of their orbital periods. It has options to show the Solar System planets and also perform a [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/mixture.html) clustering to identify 3 clusters of planets in the diagram.

<img src="https://github.com/vadibekyan/ExoPLots/assets/25388659/48aab43c-3f11-45de-880a-5f30e1658c42" alt="Figure" width="600" height="400">



Further information is given (work in progress) in the ``exoplanet_visualization.ipynb``
