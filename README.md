The following data analysis was based on data sourced from a [Kaggle competition](https://www.kaggle.com/competitions/predict-west-nile-virus/overview). It is further visualized with key insights highlighted in a [Tableau presentation](https://public.tableau.com/app/profile/jack.didier/viz/ChicagoWestNileVirus/WestNileVirusinChicago).

Its key purpose is to show the prevalence of West Nile Virus in Chicago from the years 2007-2013 during the active mosquito months of May-October and the effects that weather and pesticides have on mosquito populations.
Data includes: 
1) weather - a NOAA dataset of all weather values in the two Chicago weather stations
2) traps - a CSV file of all the times a mosquito trap was checked, where it was, how many mosquitos were there, the mosquito species, and if WNV was found present
3) spray - coordinates and times where mosquito pesticides were sprayed.

The Python code attached (West Nile Virus in Chicago.ipynb) in this repository first uploads, cleans, and organizes the data into a usable data format joining several different data sources. As the presentation shows, this data is used to identify the mosquito species Culex Pipiens as clear driver in West Nile Virus cases. It also identifies several key weather patterns including squalls that are a statistically significant indicator of both mosquito population changes and West Nile Virus prevalence. This data is further clarified by taking data from 2 weather stations and applying the correct data to whichever station each given mosquito trap is closer to. Finally, the data is used to measure the efficacy of pesticide spraying on mosquito population and WNV prevalence by both distance and time.
