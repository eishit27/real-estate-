# real-estate-

This project delivers a comprehensive machine learning model designed to predict real estate prices in Bangalore, India, providing a data-driven tool for market stakeholders. The model is built upon the "Bengaluru House Price Data" dataset from Kaggle, selected for its relevance to a major metropolitan market and its presentation of realistic data cleaning and feature engineering challenges.

The project workflow follows a robust data science pipeline. The process begins with meticulous data cleaning, where missing values are handled and inconsistent data, such as total_sqft ranges, are standardized. A critical feature engineering phase follows, which includes creating a price_per_sqft metric for analysis and performing dimensionality reduction on the high-cardinality location feature by grouping infrequent localities into an 'other' category. This enhances model stability and performance. A multi-layered outlier removal strategy is then employed, using both domain-specific rules and statistical methods to purify the dataset.

To ensure optimal performance, several regression algorithms, including Linear Regression and Random Forest, were systematically compared using GridSearchCV for hyperparameter tuning. The final selected model achieved a strong R-squared score of 0.85 on the independent test set, indicating it can explain 85% of the price variance in the data.

The trained model is serialized into a .pickle file and is accompanied by a predictor.py script. This allows users to easily load the model and obtain instant price estimations by providing key property attributes like location, square footage, BHK, and number of bathrooms, making it a practical end-to-end solution.

created by -- EISHIT JAIN (1st year CSE AI/ML) 
