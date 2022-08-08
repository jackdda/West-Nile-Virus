This data analysis was based on data sourced from a Kaggle competition:
https://www.kaggle.com/competitions/predict-west-nile-virus/code?competitionId=4366&sortBy=voteCount

It is visualized and explored further in a Tableau workbook at this link:
https://public.tableau.com/app/profile/jack.didier/viz/WestNileVirusinChicago_16591334267950/Story1

Its key purpose is to show the prevalence of West Nile Virus in Chicago from the years 2007-2013 during the active mosquito months of May-October and the effects that
weather and pesticides have on mosquito populations.
Data includes: 
1) weather - a NOAA dataset of all weather values in the two Chicago weather stations
2) traps - a CSV file of all the times a mosquito trap was checked, where it was, how many mosquitos were there, the mosquito species, and if WNV was found present
3) spray - coordinates and times where mosquito pesticides were sprayed.

The Jupyter notebook uploaded first cleans and organizes the data. Second, the Haversine formula is used to assign a closest station to each trap for more accurate
weather value pairing and individual station values are shown. From there, the CodeSum column from weather (which includes multiple 2-character weather codes on the same 
line) is split up into individual columns that can be accessed and rolling averages are performed on all values for 1-14 days the explore the effect of weather on 
mosquito growth. Third, data is visualized on a map of Chicago first showing trap locations, if they have logged WNV, and spray locations. The second map identifies 
the closest weather station to each trap. Next the data was tested for correlated values with very little evidence of correlation. Mosquito species data was then used
to perform a regression against WNV prevalence showing a clear driver in Culex Pipien. A regression was then performed on rolling averages of weather patterns showing
little effect in the first few days (with a higher coefficient on the constant than other values), but was found to be more consequential in later days. A line chart
is then shown showing statistical significance for the 3 highest coefficient values.