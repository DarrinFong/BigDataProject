# BigDataProject
SOEN 499 project repository

## Team Members
- David Berard - 40004440
- Darrin Fong - 27771959
- Loic Huss - 40000289

## Abstract
This repository will contain our work for the SOEN-499 project.
 Our project consists of a data-set analysis on the past weather,
 seasonal, pesticide, tourism, and forest industry's impact to the
 frequency and severity of forest fires in Canada.
 
 Our data-set will be drawn from the Canadian National Resource
 database. The  primary data-sets will be the location, severity
 of forest fires in Canada, daily campsite visits, past weather,
 forestry data including lumbering and pesticide use. 
 
 Our prediction is that the increase use of pesticide, lack of
 precipitations, increase of campsite usage, and increase in lumber
 harvesting will contribute on their own to the occurrence and
 intensity of forest fires.
 
## Introduction
It is widely assumed that humans are responsible for a majority of the wildfires taking place in the last years. Our hypothesis is that the use of pesticides to reduce the number of conifers in our forest, the increase of campsite usage, the increase in lumber harvesting and the lack of precipitation are contributing to the occurrence and intensity of forest fire. 

To prove our hypothesis, we will use the k-nearest neighbors (KNN) algorithm to create an interface that will evaluate the probability of a wildfire occurring based on a set of information such as weather in the last month, average campsite per kilometer square and forest composition. In parallel, we will use the random forest algorithm to predict the new location of the new occurrences. 

The two algorithms briefly described above will build the foundation for a deeper understanding of our case study. Based on those models, we will be able to determine the impact of each component and their proponents in creating an occurrence.    

## References
- The following link contains the information about the technology chosen by the team, as you can see it supports classification, regression, and clustering which supports our underlying algorithms. : https://scikit-learn.org/stable/
- More specificaly for random forest : https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
- More specificaly for k-nearest neighbors (KNN) : https://scikit-learn.org/stable/modules/neighbors.html
- The following link contains a dataset that enables us to analyse wildfire data in canada in terms of number of fires by jurisdiction, cause class, response category, and protection zone. : http://www.nfdp.ccfm.org/en/data/fires.php
