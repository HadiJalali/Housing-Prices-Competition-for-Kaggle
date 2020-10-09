# Housing-Prices-Competition-for-Kaggle
Housing Prices Competition for Kaggle Learn Users

Here I tried to use hyperparameters to examine the different values for creating a random forest purly based on [Introduction to Machine Learning course](https://www.kaggle.com/learn/intro-to-machine-learning). 

For this challenge I used 5 fold cross validation and searched across 150 different combinations. This solution obtained from this article [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74).

Using this configuration it took around an houre using 4 Ci7 cores, alghouth it does not improved that much my initial results - which, I guess, it is because of random forest limitaion.

Final result:

```python
{'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': False}
Validation MAE for Random Forest Model: 15,962
```

With this result I scored top 11 percentile.
