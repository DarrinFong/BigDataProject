# Hotel's repeat guest and its profile
### SOEN 499 : Big Data Analytics (W2020)
### Concordia University - Department of Computer-Science and Software Engineering

### Presented by:
| David Bérard - 40004440
| Darrin Fong - 27771959
| Loïc Huss - 40000298

### Presented to:
Glatard, T., Associate Professor. April 12, 2020

## Abstract
In this project, we will analyze a dataset that lists the past bookings of 2 hotels. We will use the different information in the table to determine which booking parameter has the greatest influence on whether the hotel customer is a returning customer. This can provide business insights on how to invest in marketing strategy or what is the best partnership model to bring long term value to the hotel.
Our dataset can be found on [Kaggle](https://www.kaggle.com/) under the name "[Hotel booking demand](https://www.sciencedirect.com/science/article/pii/S2352340918315191)", it originated from an article titled "Hotel booking demand datasets" from the article "[Data in Brief](https://www.sciencedirect.com/journal/data-in-brief)". The given dataset provides information as to which hotel, when the booking was made, length of stay, the number of adults, children, and/or babies, the number of required parking spaces, the meal preference, and the country of origin, among other things.
Our prediction on which booking parameters have the greatest influence on returning customers are: hotel, the market segment of the customer, and the customer type.

## Introduction
Hotels evolve in a competitive market, they are required to take innovative decisions to bring customers through the doors. They used to rely on experience and instinct to drive their initiatives but the use of big data analytics is now their first recourse. Websites such as Booking.com have extensive programs using Machine Learning and Data Analytics to enhance user experience and translate comments to any languages, among other things (See Booking.ai).
In this project, we implemented our take on such systems by applying two analysis techniques: Decision Tree and Random Forest. By implementing those techniques, we will, on one hand, answer the question "What are the features influencing most our ability to predict if a customer will become a repeat-guest?" and, on the other, provide insight on which technique provides the most accurate predictions.
The two algorithms briefly described above will build the foundation for a deeper understanding of our case study. Based on those models, we will be able to determine the impact of each component and their proponents in creating an occurrence.

## Materials and Methods

### Dataset
Our selected dataset is from the online community of data scientists and machine learning practitioners that forms Kaggle.com. We chose data that was cleaned and had a high usability index. The dataset is [Hotel booking demand](https://www.kaggle.com/jessemostipak/hotel-booking-demand), which provide extensive information on booking information. You can find information about customer provenance, type of room, type of client, length of stay and number of occupant.

### Technologies & Algorithms
As stated previously, the two algorithms selected for the evaluation are the decision tree and random forest algorithms. For both algorithms that need to be implemented for this project, Apache Spark provides all necessary methods and sufficient performance to supports classification and regression. 

The objective of those method is to create a model that provided prediction with a certain degree of accuracy. In order to evaluate their performance, we will calculate the accuracy, the precision, the recall, and finally the F1-score of both algorithms. This will provide information as which method provides the best model to predict if a customer will be a repeat-guest. And finally, both models will create a list of the three features affecting the most our prediction based on the "featureImportances" column of the model created with our two algorithms. I mean value will be evaluate to provide a list of the most influent features as per all the iteration ran.

### Data preparation
In our understanding, data preparation is separated into two distinct steps: data cleaning and transformation. In this case, the dataset we obtained was previously cleaned by [Thomas Mock and Antoine Bichat](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/readme.md) so there was no additional cleaning required. It is worth mentioning that we had to perform typecasting on the dataset’s column values once it was imported as a DateFrame, this is due to pyspark initially evaluating all values as strings when importing from a .csv file.

After the minimal data cleaning, we had to transform the data such that it can be ingested by the chosen classifiers. Firstly we had to determine which columns contain categorical values, these are the columns identified for this dataset: hotel, arrival_date_month, meal, country, market_segment, distribution_channel, reserved_room_type, assigned_room_type, deposit_type, customer_type, reservation_status. We then had to convert these categorical values into indices using pyspark’s StringIndexer. 

After the first conversion, we had to convert the values once more into one-hot vectors using pyspark’s OneHotEncoder. The reason for this conversion is because our prediction model will assume a continuous relationship between indices and potentially falsify results. For example, for a given categorical column with values A = apple, B = banana, and C = cake, StringIndexer will attribute values to categories as followed: A = 1, B = 2, and C = 3. This is acceptable for categories that have a linear relationship between them, but we can see this technique falls apart when no such relationship exists; in this case it implies the average between an apple and a cake is a banana (1+32=2). After these two conversions, the resulting numerical data columns and the One-Hot vector columns are merged into the ‘features’ column by the VectorAssembler so the data can be used in pyspark.ml.classification’s classifiers.

Figure 1: Categorical values, Index values, OneHot vector values

### Data sampling
Having the issue of data imbalance in mind, we initially sampled our entire dataset by using the randomSplit method with a ratio of 7:3 (train:test), then subsampling the majority data points to obtain a more ideal 50:50 split of is_repeated_guest:non_repeated_guest. This resulted in a loss of ~94% of the training set’s data points (from ~84k to ~5K).
After generating the model with the sampled data and obtaining less than desirable results, we realized in our case that data imbalance is less of an issue than the loss of information. This is because the goal is to determine feature importance, so this is an anomaly detection problem where the number of data points matter more than the distribution.
In order to preserve as many data points as possible, we modified our sampling method to first divide the initial dataset into two sets, one containing all rows where is_repeated_guest=True and the rest in the other. We then randomly sampled from both dataset with the predetermined split, and returned the unified results, this method allowed us to eliminate the loss of data points. It is worth mentioning we only implemented this sampling method for columns with boolean values, a more generic way would be to separate the dataset into n subsets, each containing the same value in the specified column, then randomly sample and unify them to obtain the training set and test set.
 
## Results Analysis

### Decision Tree classifier
Using the first, unchanged schema with a Decision Tree, it quickly came to our attention that the model was heavily over-specified to the column we were predicting, inherently making it useless. We decided to remove the is_repeated_guest column for the next schema.

Figure 2: DT schema 1

After running the second schema, we noticed strikingly good results including near perfect accuracy and a very good f1_score. However, when looking at the feature importances we noticed that the model was once again over-specified on a single column. This time, the model was basing itself on the previous_bookings_ not_canceled column. Logically this makes sense, since any returning guest would have to have a previous reservation which was not canceled. However, this once again made the model somewhat useless, so we decided to remove the column in question. It must be noted that the accuracy rating is skewed by the fact that the initial data has a much larger negative than positive ratio in general, and as such would have high accuracy even if the whole dataset was predicted to be negative. 
For our third schema we then removed all columns without any feature importances above 0.0001. However, this barely impacted the f1 score. In a last attempt to improve the model, we decided to adjust the maximum depth of the tree to generate ideal results, keeping in mind that too deep of a tree may result in over-specification to this schema. To avoid this, we decided on a feature importance cutoff of 0.2, which after testing gave us an ideal max depth of 10.

Figure 3: DT schema 2

Figure 4: DT schema 3

### Random Forest classifier

Figure 5: RF schema 1

Figure 6: RF schema 2

In the case of the Random Forest classifier, a similar analysis was conducted. In the first and second models we noticed similar over-specification being done to the is_repeated_ guest and previous_bookings_not_canceled columns, albeit to a lesser degree. This fits with what we expected to happen and reinforces our decision to also use random forest to act as a countermeasure to over-specification. Like what was done with the Decision Tree, for our third model we first attempted to remove columns under a certain feature importance threshold, which once again did not much impact our results. Thereafter we altered the number of trees and the max depth of each tree to improve our f1 score. The maximum f1 score we were able to achieve with our third model of data selection was with 10 trees and a max depth of 30.

Figure 7: RF schema 3

## Discussion
Analysis of our data with both of the algorithms we decided on shows us that lead_time (the number of days between when a reservation is made and the date of the stay), company (the ID of the company through which a booking is made) and adr (the nightly rate) are the most important columns in deciding whether or not a guest will be returning. These findings come as somewhat of a surprise to us. Although some hotel types and market segments were in the top 10 most important features, neither of the categories had as much of an impact as we expected, and customer type had almost no impact at all, with even the most "important" customer type holding a feature importance of barely 0.02.
If we were to decide on a single algorithm to use in the future, although our Decision Tree model seems to have better results in this case, it is generally acknowledged that they tend to be more specific than a Random Forest model, which you can see by looking at our feature importance scores. Therefore, we would argue that the less specific model generated by Random Forest would be more useful than the ever-so-slightly more accurate Decision Tree model.
Having decided that, we can focus on what the results could be used for. Given our results with this dataset, it could be possible to take it one step further and create marketing schemes for booking agencies to advertise the most optimally rated rooms to their clientele at the most ideal times of the year. 
 

## References
- "Overview - Spark 2.4.5 Documentation - Apache Spark." https://spark.apache.org/docs/latest/. Accessed 1 Apr. 2020.
- Antonio, N. and Almeida, A. and Nunes, L. (2019) ‘Hotel booking demand datasets’ in Data in Brief, Vol. 22, pp 41-49.
- "Hotel booking demand | Kaggle." 13 Feb. 2020, https://www.kaggle.com/jessemostipak/hotel-booking-demand. Accessed 1 Apr. 2020.
- "Classification and regression - RandomForestClassifier - Apache Spark." https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier. Accessed 1 Apr. 2020.
- "Classification and regression - DecisionTreeClassifier - Apache Spark." https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier. Accessed 1 Apr. 2020.
- "StringIndexer - Apache Spark." https://spark.apache.org/docs/latest/ml-features. Accessed 1 Apr. 2020.
- "Extracting, transforming and selecting features - OneHotEncoder - Apache Spark." https://spark.apache.org/docs/latest/ml-features#onehotencoder. Accessed 1 Apr. 2020. 


