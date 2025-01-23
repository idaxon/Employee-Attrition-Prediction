# Employee Attrition Prediction

## Overview

This project predicts employee attrition using machine learning techniques, specifically a Random Forest Classifier. The model uses workplace factors to predict whether an employee will leave the company. The application provides an interactive dashboard built using Dash and Plotly for visualization.

## Features

- **Employee Attrition Prediction:** Predicts whether an employee will leave the company based on various factors such as job role, age, etc.
- **Model Accuracy:** Displays the accuracy of the Random Forest model used for prediction.
- **Feature Importance Visualization:** Shows a bar chart indicating the most important features contributing to employee attrition.
- **Attrition Rate Distribution:** A pie chart to visualize the distribution of employee attrition status (Yes/No).
- **Employee Information Table:** A table listing employees along with their predicted attrition status.
- **Prediction for New Employees:** Allows users to input new employee data and predict their attrition likelihood.



## Technologies Used

- **Python:** For data processing and machine learning model.
- **Dash:** For building the interactive dashboard.
- **Plotly:** For visualizations such as bar charts, pie charts, and histograms.
- **Scikit-learn:** For machine learning algorithms and model evaluation.
- **Pandas & NumPy:** For data manipulation and processing.

## Dataset

The dataset used for this project is [EmployeeAttrition.csv](https://raw.githubusercontent.com/ybifoundation/Dataset/main/EmployeeAttrition.csv), which contains information about employees, including their demographic details, job role, and attrition status.

## How It Works

1. **Data Preprocessing:**
   - The dataset is cleaned by converting categorical variables into numerical values using one-hot encoding.
   - The `Attrition` column is converted into a binary format (`Yes` -> 1, `No` -> 0).

2. **Model Training:**
   - A Random Forest Classifier is trained on the data to predict employee attrition.
   - Model accuracy is computed and displayed on the dashboard.

3. **Interactive Dashboard:**
   - The dashboard allows users to visualize:
     - Model accuracy.
     - Feature importance.
     - Attrition rate distribution.
     - Employee details with their predicted attrition status.
     - Predictions for new employee data.

## Example Use Case

1. **Prediction for New Employee:**
   - Input basic details such as age and job role for a new employee.
   - The model predicts whether the employee is likely to leave the company.

2. **Feature Importance:**
   - The model highlights the key features (e.g., age, job role, distance from home) that impact the prediction of attrition.


## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request with your changes. Ensure that your contributions align with the overall project goals and maintain code quality.




