# Hotel's repeat guest and its profile
David Bérard - 40004440

Darrin Fong - 27771959 

Loïc Huss - 40000298 

## Abstract
In this project, we will analyze a data-set that lists the past bookings of 2 hotels. We will use the different information in the table to determine which component has the greatest influence in determining the profile of the customer who is most likely to come back. This can provide business insights on how to invest in marketing strategy or what is your best partnership model to bring long term value to the hotel.

Our data-set can be found on [Kaggle](https://www.kaggle.com/) under the name "Hotel booking demand". The given data-sets provide information as to which hotel, when the booking was made, length of stay, the number of adults, children, and/or babies, the number of required parking spaces, the meal preference, and the country of origin, among other things.

Our prediction is that which hotel, the market segment of the customer and the customer type will be the three main drivers to determine if a guest is going to be a repeat guest.

## Introduction

Hotels evolve in a competitive market, there are required to take innovative decision to bring customers through the doors. They use to rely on experience and instinct to drive their initiatives but the use of big data analytics is now their first recourse. Websites such as [Booking.com](https://www.booking.com/) have extensive programs using Machine Learning and Data Analytics to enhance user experience and translate comments to any languages, among other things (See [Booking.ai](https://booking.ai/)).

In this project, we implemented our take on such systems by applying two analysis techniques: Decision Tree and Random Forest. By implementing those techniques, we will, on one hand, answer the question "What are the features influencing most our ability to predict if a customer will become a repeat-guest?" and, on the other, provide insight on which technique provides the most accurate predictions.

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
- "Overview - Spark 2.4.5 Documentation - Apache Spark." https://spark.apache.org/docs/latest/. Accessed 1 Apr. 2020. Antonio, N. and Almeida, A. and Nunes, L. (2019) ‘Hotel booking demand datasets’ in Data in Brief, Vol. 22, pp 41-49. 
- "Hotel booking demand | Kaggle." 13 Feb. 2020, https://www.kaggle.com/jessemostipak/hotel-booking-demand. Accessed 1 Apr. 2020.
- "Classification and regression - RandomForestClassifier - Apache Spark." https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier. Accessed 1 Apr. 2020.
- "Classification and regression - DecisionTreeClassifier - Apache Spark." https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier. Accessed 1 Apr. 2020.
- "StringIndexer - Apache Spark." https://spark.apache.org/docs/latest/ml-features. Accessed 1 Apr. 2020.
- "Extracting, transforming and selecting features - OneHotEncoder - Apache Spark." https://spark.apache.org/docs/latest/ml-features#onehotencoder. Accessed 1 Apr. 2020. 
