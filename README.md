# Wildfires and its causes
David Bérard - 40004440

Darrin Fong - 27771959 

Loïc Huss - 40000298 

## Abstract
Our project consists of a data-set analysis on the past weather, seasonal, pesticide, tourism, and forest industry's impact to the frequency and severity of forest fires in Canada.

Our data-set will be drawn from the Canadian National Resource database. The primary data-sets will be the location, severity of forest fires in Canada, daily campsite visits, past weather, forestry data including lumbering and pesticide use.

Our prediction is that the increase use of pesticide, lack of precipitations, increase of campsite usage, and increase in lumber harvesting will contribute on their own to the occurrence and intensity of forest fires.

## Introduction
It is widely assumed that humans are responsible for a majority of the wildfires taking place in the last years. Our hypothesis is that the use of pesticides to reduce the number of conifers in our forest, the increase of campsite usage, the increase in lumber harvesting and the lack of precipitation are contributing to the occurrence and intensity of forest fire. 

To prove our hypothesis, we will use the k-nearest neighbors (KNN) algorithm to create an interface that will evaluate the probability of a wildfire occurring based on a set of information such as weather in the last month, average campsite per kilometer square and forest composition. In parallel, we will use the random forest algorithm to predict the new location of the new occurrences. 

The two algorithms briefly described above will build the foundation for a deeper understanding of our case study. Based on those models, we will be able to determine the impact of each component and their proponents in creating an occurrence.   

## Materials and Methods

As stated previously, the two algorithms selected for the evaluation are the k-nearest neighbors (KNN) and random forest algorithms. For both algorithms that need to be implemented for this project, Scikit-Learn seems to be the better choice as it supports classification, regression, and clustering which greatly simplifies the implementation of the k-nearest neighbors (KNN) and random forest algorithms. We plan on using the adaptive micro-cluster nearest neighbour data classification method (MC-NN) to efficiently determine campsites or forest areas that are likely to host a forest fire, with the help of a constant weathers station data stream. Each subset of our model will represent zones with a constant value representing likelihood of wildfires.


As for datasets, our primary dataset will be the [National Forestry Database](http://www.nfdp.ccfm.org/en/data/fires.php), which will give us access to information about canadian forest fires by month, jurisdiction (province), cause and size amongst other classifications.
For campsite attendance, so far we only have [Parks Canada](https://www.pc.gc.ca/en/docs/pc/attend/table2)'s yearly attendance statistics, which do not provide enough temporal data to satisfy our needs. We will continue searching for datasets for this matter, but in case it is not possible we will look for a single region’s monthly data and assuming all regions have a similar attendance percentage by month.
Climate data can be acquired on the government's [climate website](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html), and will be used to distinguish the most frequent type of weather/temperatures in the area of a fire.
The pesticide usage data will be from the [Pest Management Regulatory Agency (PMRA) - Pest Control Product Registrant Inspections Database](https://open.canada.ca/data/en/dataset/5d379500-64ab-4bc4-9b4e-ab6f9cbcc695). We will be assuming the pest control product registrant inspections location are synonymous with the pesticide usage location.
For all of the databases mentioned above, we will have to get the latitude and longitude of the city from another source as the databases only supply the name of the regions from where the data originates, specifically the weather data. 

## Type of project
Dataset analysis: select a dataset (for instance from your research) and apply at least two techniques seen in the course using Apache Spark, Dask or scikit-learn. You are not required to re-implement these techniques, but you need to discuss and interpret the results.

## References
- The following link contains the information about the technology chosen by the team, as you can see it supports classification, regression, and clustering which supports our underlying algorithms. : https://scikit-learn.org/stable/
- More specifically for random forest : https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
- More specifically for k-nearest neighbors (KNN) : https://scikit-learn.org/stable/modules/neighbors.html
- The following link contains a dataset that enables us to analyse wildfire data in canada in terms of number of fires by jurisdiction, cause class, response category, and protection zone. : http://www.nfdp.ccfm.org/en/data/fires.php
