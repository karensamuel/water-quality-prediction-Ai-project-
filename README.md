# Water Quality Prediction Project

## Objective
The primary objective of this project was to develop a predictive model to determine water quality, specifically distinguishing between potable (safe to drink) and non-potable water. This project was a part of our second-year AI course.

## Data Preprocessing and Cleaning
1. **Data Source**: The water quality dataset was provided by our college and included various chemical and physical parameters.
2. **Data Cleaning**: Addressed missing values through imputation techniques, handled outliers, and corrected any inconsistencies in the dataset.

## Handling Class Imbalance
- **Random Oversampling**: To address the class imbalance issue (as potable water instances were significantly fewer than non-potable), we used random oversampling to balance the dataset by duplicating samples from the minority class.

## Model Training
We trained and evaluated six different machine learning models to predict water quality:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. Naive Bayes
6. k-Nearest Neighbors (k-NN)

## Hyperparameter Tuning
- **Grid Search**: Employed grid search with cross-validation to find the optimal hyperparameters for each model, enhancing their performance by exhaustively searching through predefined parameter grids.

## Model Evaluation
- Compared the models based on accuracy, precision, recall, F1-score, and ROC-AUC to select the best-performing model.
- The Random Forest model emerged as the top performer, providing a good balance between bias and variance, along with robust predictive performance.

## Deployment with Streamlit
- **User Interface**: Developed a user-friendly GUI using Streamlit, allowing users to input water quality parameters and get instant predictions on water potability.
- **Real-time Prediction**: The Streamlit app integrated the trained model to provide real-time predictions, making it accessible for non-technical stakeholders to assess water quality quickly and easily.

## Conclusion
This project demonstrated a comprehensive approach to solving the water quality prediction problem. By combining robust data preprocessing, addressing class imbalance with random oversampling, evaluating multiple models, and fine-tuning them with grid search, we developed a reliable and accessible water quality prediction system. The deployment through Streamlit ensured that the solution was user-friendly and could be utilized effectively by stakeholders for real-world applications.
