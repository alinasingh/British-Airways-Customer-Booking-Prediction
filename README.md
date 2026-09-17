# British-Airways-Customer-Booking-Prediction
A machine learning project to predict customer booking completion for British Airways using customer booking data, exploratory data analysis, feature engineering, and a Random Forest classifier.

### `Introduction`
This project focuses on analyzing customer booking data from British Airways and building a machine learning model to predict whether a customer will complete a flight booking.

The project was completed as part of the British Airways Data Science job simulation on Forage, where the objective was to use real-world customer booking data to identify patterns in customer behavior and build a predictive model.

The analysis covers data cleaning, exploratory data analysis (EDA), feature preparation, and machine learning. A Random Forest Classifier was developed to predict the booking_complete target variable and evaluate how well customer and flight-related features can help identify completed bookings.

Through this project, I explored factors such as purchase lead time, flight hour, length of stay, booking origin, flight route, travel preferences, and additional services to better understand customer booking behavior.

### `Problem Statement`
The project addresses the challenge of predicting whether a customer will complete their British Airways flight booking.

- Analyze customer booking data to understand booking behavior.
- Identify factors that may influence whether a booking is completed.
- Explore customer, flight, and booking-related features.
- Predict the booking_complete outcome using machine learning.
- Evaluate how effectively the model identifies completed bookings.
- Generate insights that can help better understand customer booking patterns.

### `Dataset Used`
<a href="https://github.com/alinasingh/British-Airways-Customer-Booking-Prediction/blob/main/customer_booking.csv"> British Airways customer booking datasets

## 🎯 Project Objective

The main objective of this project is to analyze customer booking behavior and build a Machine Learning model that can predict whether a customer will complete a flight booking.

The project aims to:

* Understand customer booking patterns and behaviors.
* Explore factors associated with booking completion.
* Perform data cleaning and preprocessing.
* Analyze categorical and numerical features through EDA.
* Transform the data into a format suitable for Machine Learning.
* Build a **Random Forest Classifier** to predict `booking_complete`.
* Evaluate the model using metrics such as **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
* Identify the most important features influencing the model's predictions.
* Generate meaningful business insights from the analysis.

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand customer booking behavior and identify patterns associated with completed bookings.

### 1. Booking Completion Distribution

The target variable `booking_complete` is imbalanced:

* **Not Completed:** 42,522 customers — 85.00%
* **Completed:** 7,478 customers — 15.00%

This shows that only a smaller proportion of customers in the dataset completed their bookings.

### 2. Sales Channel

Customers using the **Internet channel** had a higher booking completion rate than customers using the Mobile channel.

* **Internet:** 15.49%
* **Mobile:** 10.98%

This indicates a noticeable difference in booking completion behavior between the two sales channels.

### 3. Trip Type

Booking completion varied across different trip types:

* **Round Trip:** 15.10%
* **One Way:** 5.18%
* **Circle Trip:** 4.31%

Round-trip bookings represented the highest completion rate among the three trip types.

### 4. Booking Origin

The analysis showed substantial differences in booking completion rates across customer booking origins.

Some of the major booking origins were:

| Booking Origin | Customers | Booking Completion Rate |
| -------------- | --------: | ----------------------: |
| Australia      |    17,691 |                   5.06% |
| Malaysia       |     7,055 |                  34.50% |
| South Korea    |     4,502 |                  10.20% |
| Japan          |     3,819 |                  12.36% |
| China          |     3,284 |                  20.83% |

Malaysia had a considerably higher completion rate compared with several other major booking origins.

### 5. Flight Routes

Booking completion also varied significantly across flight routes.

Some of the most frequent routes included:

| Route  | Customers | Booking Completion Rate |
| ------ | --------: | ----------------------: |
| AKLKUL |     2,620 |                  21.34% |
| PENTPE |       912 |                  43.42% |
| MELSGN |       833 |                   5.16% |
| ICNSIN |       793 |                  11.22% |
| DMKKIX |       729 |                  25.38% |

The analysis demonstrates that customer booking behavior differs considerably across routes.

### 6. Additional Services

Customers requesting additional services showed different booking completion rates:

| Additional Service    | Completion Rate |
| --------------------- | --------------: |
| Extra Baggage – Yes   |          16.72% |
| Extra Baggage – No    |          11.53% |
| Preferred Seat – Yes  |          17.76% |
| Preferred Seat – No   |          13.84% |
| In-flight Meals – Yes |          16.09% |
| In-flight Meals – No  |          14.18% |

Customers who selected additional services generally showed higher booking completion rates in this dataset.

### 7. Flight Duration

The average flight duration was approximately **7.28 hours**.

* Completed bookings: **6.90 hours**
* Not completed bookings: **7.35 hours**

This suggests some variation in booking completion behavior based on flight duration.

### Key EDA Findings

The exploratory analysis highlighted several variables associated with differences in booking completion rates, including:

* Sales channel
* Trip type
* Booking origin
* Flight route
* Purchase lead time
* Length of stay
* Flight hour
* Additional services
* Flight duration

These findings were used to prepare the data for Machine Learning and to better understand the factors captured by the predictive model.

## ⚙️ Data Preprocessing & Feature Engineering

Before building the Machine Learning model, the dataset was cleaned and transformed into a suitable format for modeling.

### Data Preprocessing Steps

* Checked the dataset structure and data types.
* Checked for missing values.
* Removed duplicate records.
* Separated the target variable `booking_complete` from the input features.
* Identified categorical and numerical features.
* Applied **One-Hot Encoding** to categorical variables.
* Split the dataset into training and testing sets.
* Prepared the transformed features for Machine Learning.

### Train-Test Split

The cleaned dataset was divided into:

* **Training Set:** 39,424 records
* **Testing Set:** 9,857 records

After One-Hot Encoding, the feature matrix contained **901 features**.

### Target Variable

The target variable was:

`booking_complete`

where:

* `0` → Booking was not completed
* `1` → Booking was completed

This preprocessing ensured that both categorical and numerical information could be effectively used by the Machine Learning model.


## 🤖 Machine Learning Model

After completing the data preprocessing and feature transformation steps, a **Random Forest Classifier** was used to predict whether a customer would complete their flight booking.

### Why Random Forest?

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to make predictions. It can handle both numerical and encoded categorical features and is also useful for identifying the relative importance of features.

### Model Configuration

The Random Forest model was created with the following parameters:

```python
RandomForestClassifier(
    n_estimators=200,
    criterion='gini',
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
```

### Model Training

The model was trained using the transformed training dataset and then evaluated on the unseen test dataset.

The evaluation focused on:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* 5-Fold Cross-Validation

The model's probability predictions were also used to calculate the **ROC-AUC score**, which provides a useful measure of how well the model distinguishes between completed and non-completed bookings.


## 📈 Model Performance & Evaluation

The Random Forest model was evaluated on the test dataset using multiple classification metrics.

### Test Set Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **84.86%** |
| Precision | **47.90%** |
| Recall    | **10.83%** |
| F1-Score  | **17.66%** |
| ROC-AUC   | **0.7702** |

### Confusion Matrix

The model produced the following confusion matrix:

```text
[[8205,  174],
 [1318,  160]]
```

This indicates that the model correctly identified a large number of non-completed bookings, while detecting fewer of the completed bookings.

### Cross-Validation

To check the consistency of the model, **5-Fold Cross-Validation** was performed using ROC-AUC.

* **Mean CV ROC-AUC:** 0.7783
* **Fold Scores:** approximately 0.774 – 0.782

The cross-validation results were relatively close across the five folds, providing an additional view of the model's performance across different subsets of the data.

### Model Interpretation

Although the model achieved an accuracy of **84.86%**, accuracy alone does not fully describe performance because the target variable is imbalanced, with completed bookings representing only about 15% of the dataset.

Therefore, **Precision, Recall, F1-Score, and ROC-AUC** were also considered when evaluating the model.


## ⭐ Feature Importance

Feature importance was analyzed from the trained Random Forest model to understand which variables contributed most to the model's predictions.

### Top Important Features

| Rank | Feature                    | Importance |
| ---: | -------------------------- | ---------: |
|    1 | `purchase_lead`            |     0.1455 |
|    2 | `flight_hour`              |     0.1218 |
|    3 | `length_of_stay`           |     0.1104 |
|    4 | `num_passengers`           |     0.0472 |
|    5 | `flight_duration`          |     0.0364 |
|    6 | `booking_origin_Malaysia`  |     0.0309 |
|    7 | `wants_in_flight_meals`    |     0.0232 |
|    8 | `booking_origin_Australia` |     0.0211 |
|    9 | `wants_preferred_seat`     |     0.0182 |

### Key Observations

The model assigned the highest feature importance to:

* **Purchase Lead** — the number of days between booking and flight.
* **Flight Hour** — the scheduled departure hour.
* **Length of Stay** — the number of days spent at the destination.
* **Number of Passengers** — the number of passengers included in the booking.
* **Flight Duration** — the duration of the flight.

Among the encoded categorical variables, **booking origin** and customers' preferences for additional services also contributed to the model's predictions.

> **Note:** Feature importance indicates how much a feature contributes to the Random Forest's predictive decisions. It does not by itself establish that a feature causes a booking to be completed.


## 💡 Key Business Insights

The analysis provided several insights into customer booking behavior:

* **Booking lead time** was the most important feature in the Random Forest model, indicating that the timing of a booking was strongly associated with the model's predictions.

* **Flight timing** was another important factor, with `flight_hour` ranking among the top features in the model.

* **Length of stay** also had a high feature importance, suggesting that trip duration captured useful information about booking behavior.

* **Sales channel** showed different booking completion rates. Internet bookings had a higher completion rate (**15.49%**) compared with Mobile bookings (**10.98%**) in the analyzed dataset.

* **Trip type** showed noticeable differences in booking completion rates. Round-trip bookings had a completion rate of **15.10%**, compared with **5.18%** for One Way and **4.31%** for Circle Trip bookings.

* **Booking origin and route** were associated with substantial differences in booking completion rates. For example, Malaysia had a completion rate of **34.50%**, while Australia had **5.06%** among the major booking origins analyzed.

* **Additional services** were associated with higher booking completion rates in this dataset. Customers requesting preferred seats, extra baggage, or in-flight meals showed higher completion rates than customers who did not request those services.

* The model's feature importance analysis highlighted **purchase lead, flight hour, length of stay, number of passengers, and flight duration** among the most influential features.

These findings can help provide a better understanding of customer booking patterns and can serve as a starting point for further analysis of customer behavior.


## 🏁 Conclusion

This project analyzed British Airways customer booking data to understand booking behavior and predict whether a customer would complete a flight booking.

The analysis included data cleaning, exploratory data analysis, categorical feature encoding, train-test splitting, and Machine Learning model development. A **Random Forest Classifier** was used for prediction.

The model achieved an **accuracy of 84.86%** and a **ROC-AUC score of 0.7702** on the test dataset. The 5-fold cross-validation produced a mean ROC-AUC of approximately **0.7783**.

The feature importance analysis identified **purchase lead, flight hour, length of stay, number of passengers, and flight duration** as some of the most influential features in the model.

The project also highlighted the importance of looking beyond accuracy when working with an imbalanced classification problem. Metrics such as Precision, Recall, F1-Score, and ROC-AUC provided a more complete view of model performance.

Overall, this project provided practical experience in applying an end-to-end Data Science workflow to a real-world customer booking problem and demonstrated how Machine Learning can be used to identify patterns in customer behavior.


## 📊 Project Presentation


A detailed presentation covering the complete project workflow, including the business problem, dataset, exploratory data analysis, customer booking insights, Machine Learning approach, model performance, feature importance, and conclusion.

📎 **[View / Download Project Presentation](./British_Airways_Customer_Booking_Prediction_Presentation.pptx)**

## 🏆 Certificate

This project was completed as part of the **British Airways Data Science Virtual Experience Program on Forage**.

The program provided practical experience in applying Data Science and Machine Learning techniques to real-world business problems, including:

* Modeling lounge eligibility at Heathrow Terminal 3
* Predicting customer buying behaviour

📜 <a href="https://github.com/alinasingh/British-Airways-Customer-Booking-Prediction/blob/main/Forage%20certificate%20of%20complietion.pdf">View Certificate








