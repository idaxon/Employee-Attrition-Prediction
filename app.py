import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_table

# Step 1: Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/ybifoundation/Dataset/main/EmployeeAttrition.csv")

# Step 2: Data Preprocessing
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})  # Convert target to binary
df['EmployeeID'] = df['EmployeeNumber']  # Use EmployeeNumber as EmployeeID
df['EmployeeName'] = df['EmployeeID'].astype(str)  # Use EmployeeID as placeholder for Name

# Drop columns that aren't useful for the prediction model
X = df.drop(['Attrition', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
y = df['Attrition']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 6: Add prediction probabilities for employees
df['AttritionPrediction'] = model.predict(X)
df['AttritionPrediction'] = df['AttritionPrediction'].map({0: 'No', 1: 'Yes'})  # Convert to readable format

# Feature importance for visualization
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Step 7: Build the Dashboard
app = dash.Dash(__name__)

# Dashboard Layout
app.layout = html.Div([
    # Navbar
    html.Div([
        html.Nav([
            html.Div([
                html.H2("Employee Attrition Prediction", style={'color': '#fff', 'font-size': '15px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'padding': '0px 10px'}),
        ], style={'background-color': '#333', 'position': 'relative', 'top': '0', 'width': '100%', 'padding': '10px','height':'25px'}),
    ]),

  # Landing Page (with background and key points)
html.Div([
    html.Div([
        html.H1("Predict Employee Attrition with AI", style={'color': '#fff', 'text-align': 'center', 'font-size': '50px'}),
        html.P("Accurately predict employee attrition based on various workplace factors using machine learning algorithms.",
               style={'color': '#fff', 'text-align': 'center', 'font-size': '20px'}),
        html.H3(f"Model Accuracy: {accuracy:.2%}", style={'color': '#ffcc00', 'text-align': 'center', 'font-size': '40px'}),
        
        # Key Points Section within the Landing Page
        html.Div([
            html.H3("Key Points to Consider", style={'text-align': 'left', 'color': '#4CAF50', 'font-size': '30px'}),
            html.Ul([
                html.Li("Predict whether an employee will leave the company."),
                html.Li("Gain insights into factors affecting employee retention."),
                html.Li("Utilize machine learning to enhance HR decision-making."),
            ], style={'font-size': '18px', 'text-align': 'left', 'color': '#fff'}),
        ], style={'padding': '20px', 'border-radius': '8px', 'margin-top': '20px'}),
    ], style={'background-image': 'url("https://png.pngtree.com/thumb_back/fw800/background/20231219/pngtree-urban-business-technology-building-blue-simple-annual-meeting-celebration-image_15523203.png")', 
              'background-size': 'cover', 'height': '87vh', 'padding': '30px'}),
], style={'background-color': '#4CAF50', 'height': 'auto'}),


    # Feature Importance Chart
    html.Div([
        html.H3("Feature Importance", style={'text-align': 'center', 'color': '#4CAF50'}),
        dcc.Graph(
            figure=px.bar(feature_importances, x='Importance', y='Feature', orientation='h',
                          title="Impact of Features on Attrition",
                          labels={'Feature': 'Features', 'Importance': 'Importance'},
                          template='plotly_white',
                          height=500)
        )
    ], style={'margin-top': '30px', 'padding': '20px', 'background-color': '#f9f9f9',
              'border-radius': '8px', 'box-shadow': '2px 2px 5px #ddd'}),

    # Attrition Rate Pie Chart
    html.Div([
        html.H3("Attrition Rate", style={'text-align': 'center', 'color': '#4CAF50'}),
        dcc.Graph(
            figure=px.pie(df, names='Attrition', title="Employee Attrition Distribution", 
                          color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'}, 
                          labels={'Attrition': 'Attrition Status'}, 
                          hole=0.3)
        )
    ], style={'margin-top': '30px', 'padding': '20px', 'background-color': '#f9f9f9',
              'border-radius': '8px', 'box-shadow': '2px 2px 5px #ddd'}),

    # Employee Attrition Data Table
    html.Div([
        html.H3("Employee Attrition Information", style={'text-align': 'center', 'color': '#4CAF50'}),
        dash_table.DataTable(
            id='employee-table',
            columns=[
                {'name': 'Employee ID', 'id': 'EmployeeID'},
                {'name': 'Employee Name', 'id': 'EmployeeName'},
                {'name': 'Attrition Prediction', 'id': 'AttritionPrediction'}
            ],
            data=df[['EmployeeID', 'EmployeeName', 'AttritionPrediction']].to_dict('records'),
            style_table={'height': '350px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '10px', 'fontSize': 14},
            style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
        )
    ], style={'margin-top': '30px', 'padding': '20px', 'background-color': '#ffffff',
              'border-radius': '8px', 'box-shadow': '2px 2px 5px #ddd'}),

    # Prediction for Employee Type
    html.Div([
        html.H3("Prediction for Employee Type", style={'text-align': 'center', 'color': '#4CAF50'}),
        dcc.Graph(
            figure=px.histogram(df, x='JobRole', color='Attrition', title="Attrition by Job Role",
                                labels={'JobRole': 'Job Role', 'Attrition': 'Attrition Status'},
                                color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'})
        )
    ], style={'margin-top': '30px', 'padding': '20px', 'background-color': '#f9f9f9',
              'border-radius': '8px', 'box-shadow': '2px 2px 5px #ddd'}),

    # Employee Selection Dropdown and Prediction
    html.Div([
        html.H3("Select Employee to Predict Attrition", style={'text-align': 'center', 'color': '#4CAF50'}),
        dcc.Dropdown(
            id='employee-dropdown',
            options=[{'label': f"ID: {emp_id} - Name: {name}", 'value': emp_id}
                     for emp_id, name in zip(df['EmployeeID'], df['EmployeeName'])],
            value=df['EmployeeID'].iloc[0],  # Default selection
            style={'width': '50%', 'margin': 'auto'}
        ),
        html.Div(id='employee-details', style={'text-align': 'center', 'margin-top': '20px', 'font-size': '18px'})
    ], style={'margin-top': '40px'}),

    # Prediction Input Form for New Employee Data
    html.Div([
        html.H3("Input Data for New Employee", style={'text-align': 'center', 'color': '#4CAF50'}),
        html.Div([
            html.Label("Age:"),
            dcc.Input(id='input-age', type='number', value=30, style={'width': '50%'}),
        ], style={'margin': '10px 0'}),
        html.Div([
            html.Label("Job Role:"),
            dcc.Dropdown(id='input-job-role', options=[{'label': role, 'value': role} for role in df['JobRole'].unique()],
                         value=df['JobRole'].iloc[0], style={'width': '50%'}),
        ], style={'margin': '10px 0'}),
        html.Button('Predict Attrition', id='predict-button', n_clicks=0, style={'margin-top': '20px'}),
        html.Div(id='prediction-output', style={'text-align': 'center', 'margin-top': '20px', 'font-size': '18px'})
    ], style={'margin-top': '40px', 'padding': '20px', 'background-color': '#f9f9f9', 'border-radius': '8px'}),

    # Footer
    html.Div([
        html.P("© 2025 Employee Attrition Prediction - All Rights Reserved.", style={'text-align': 'center', 'color': '#555'}),
    ], style={'padding': '20px', 'background-color': '#f1f1f1', 'margin-top': '30px'}),

])

# Step 8: Callbacks for Interactivity
@app.callback(
    Output('employee-details', 'children'),
    [Input('employee-dropdown', 'value')]
)
def update_employee_details(emp_id):
    employee = df[df['EmployeeID'] == emp_id].iloc[0]
    prediction = employee['AttritionPrediction']
    return f"Employee ID: {emp_id}, Name: {employee['EmployeeName']}, Attrition Prediction: {prediction}"

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('input-age', 'value'), Input('input-job-role', 'value')]
)
def predict_new_employee(n_clicks, age, job_role):
    if n_clicks > 0:
        # Prepare the new data based on one-hot encoding and match the column structure
        new_data = pd.DataFrame({
            'Age': [age],
            'JobRole_Sales Executive': [1 if job_role == 'Sales Executive' else 0],
            'JobRole_Research Scientist': [1 if job_role == 'Research Scientist' else 0],
            # Add the other features accordingly
        })

        # Add missing columns from the training data (with default values of 0)
        for col in X.columns:
            if col not in new_data.columns:
                new_data[col] = 0

        # Ensure the order of columns is the same as the training data
        new_data = new_data[X.columns]

        # Make prediction
        prediction = model.predict(new_data)
        return f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}"
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
