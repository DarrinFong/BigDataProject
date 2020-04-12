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

Our selected dataset is from the online community of data scientists and machine learning practitioners that forms Kaggle.com. We chose data that was cleaned and had a high usability index. The dataset is Hotel booking demand, which provide extensive information on booking information. You can find information about customer provenance, type of room, type of client, length of stay and number of occupant. 
Technologies & Algorithms

As stated previously, the two algorithms selected for the evaluation are the decision tree and random forest algorithms. For both algorithms that need to be implemented for this project, Apache Spark provides all necessary methods and sufficient performance to supports classification and regression. 

The objective of those method is to create a model that provided prediction with a certain degree of accuracy. In order to evaluate their performance, we will calculate the accuracy, the precision, the recall, and finally the F1-score of both algorithms. This will provide information as which method provides the best model to predict if a customer will be a repeat-guest. And finally, both models will create a list of the three features affecting the most our prediction. I mean value will be evaluate to provide a list of the most influent features as per all the iteration ran.

## Type of project
Dataset analysis: select a dataset (for instance from your research) and apply at least two techniques seen in the course using Apache Spark, Dask or scikit-learn. You are not required to re-implement these techniques, but you need to discuss and interpret the results.

## References
- "Overview - Spark 2.4.5 Documentation - Apache Spark." https://spark.apache.org/docs/latest/. Accessed 1 Apr. 2020. Antonio, N. and Almeida, A. and Nunes, L. (2019) ‘Hotel booking demand datasets’ in Data in Brief, Vol. 22, pp 41-49. 
- "Hotel booking demand | Kaggle." 13 Feb. 2020, https://www.kaggle.com/jessemostipak/hotel-booking-demand. Accessed 1 Apr. 2020.
- "Classification and regression - RandomForestClassifier - Apache Spark." https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier. Accessed 1 Apr. 2020.
- "Classification and regression - DecisionTreeClassifier - Apache Spark." https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier. Accessed 1 Apr. 2020.
- "StringIndexer - Apache Spark." https://spark.apache.org/docs/latest/ml-features. Accessed 1 Apr. 2020.
- "Extracting, transforming and selecting features - OneHotEncoder - Apache Spark." https://spark.apache.org/docs/latest/ml-features#onehotencoder. Accessed 1 Apr. 2020. 
