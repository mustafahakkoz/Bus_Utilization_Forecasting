# Bus Utilization Forecasting

[BusUtilization](https://www.kaggle.com/hakkoz/piworksbus) is a toy dataset which consists hourly usage measurements for 10 municipalities. A goal of the project is implementing feature engineering methods for a time series and compare the results with several approaches (statistical, machine learning and deep learning).

- training dataset ( 418.5 kB, 13.1K rows, 4 cols)

- testing dataset ( last 2 weeks of 10 in training set can be used as test data)

Implementation details can be found in the notebook.

---

#### Online Notebooks:

1. [PIworks | BusUtilizationForecasting](https://www.kaggle.com/hakkoz/piworks-busutilizationforecasting)

---

#### Repo Content and Implementation Steps:

[**piworks-busutilizationforecasting.ipynb**](https://github.com/mustafahakkoz/Predict_IPP_Contribution/blob/main/1.garanti-eda-preprocessing.ipynb)

- Missing values, seasonalities and trends are examined to choose a proper municipality.

- An advanced feature engineering step is implemented by creating 50 new features based on a single univarite data.
  **Date-related features:** month, day, week_of_year, weekday.
  **Features based on adjacency:** hour-to-hour, day-to-day, week-to-week differences including percentage change.
  **Crossover strategy:** 18-ewm (represents daily sequences) 96-ewm (represents weekly sequences), difference of both and markings on crossover points to represent buy/sell decisions.
  **Bollinger bands:** 18-sma (simple moving average) and +- 2*std is calculated to represent upper and lower bands. Also outlier points are detected to represent enter/exit decisions.
  **Features based on time window:** Statistical measures on previous 18 timestamps are calculated. avg absolute diff, min, max, max-min diff, median, median of differences, interquartile range, values above mean, number of peaks, skewness, kurtosis, energy (average of squares).
  **Fourier Features of time window:** 12 Statistical measures defined above is calculated on frequency domain of 18-sma sequences.
  **Indices of significant points:** Significant points of short-term (1-day) past window is found. argmax, argmin, difference of both and fourier versions of them are calculated.

- 2 regression models with these features including tuning steps (Ridge and XGBoost regressors) are implemented.

- 2 statistical models SARIMAX (with gridsearch) and Prophet is implemented.

- And finally a deep learning approach (LSTM) is implemented.

---

#### Notes:

- Best score is the baseline model Ridge Regressor since we did heavy feature engineering and it works very well. It is possible to improve this model with handling overfitting by dimensionality reduction.

- Ridge classifier is even better than XGBoost and ARIMA. Because it behaves like already autoregressive and moving average model by calculating sum of statistics of lagged values.

- Generally Prophet is better than SARIMAX because it handles seasonality and outliers automatically but in our case the opposite is true owing to the tuning of SARIMAX by gridsearch.

- It is possible to improve SARIMAX by using exagenous (features) variables. Also, extending search spaces for parameters and examining ACF and PACF plots may work.

- LSTM is worst model because it's not tuned and probably sequence length is not enough.

- Implementing dynamic predictions is the scientifically right way but we choose static one to compare models to baseline model. (Though, it's possible to implement baseline model in a dynamic way.)
