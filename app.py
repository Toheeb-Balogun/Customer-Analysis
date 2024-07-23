# Import the libraries needed
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
path = r"C:\Users\bkayo\OneDrive\Documents\GitHub\Churn-Analysis GitHub Project\sampledataset (1).csv"
df = pd.read_csv(path)

# Preprocess the data
# Fill the missing values with mean
df['age'] = df['age'].fillna(df['age'].mean())
df['income'] = df['income'].fillna(df['income'].mean())
df['purchase_amount'] = df['purchase_amount'].fillna(df['purchase_amount'].mean())
# Convert the purchase_date to a datetime
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
# Extract the year and month from the purchase_date
df['purchase_year'] = df['purchase_date'].dt.year
df['purchase_month'] = df['purchase_date'].dt.month
# drop id
df = df.drop(columns = ['name', 'id'])
### Encode Categorical columns
df = pd.get_dummies(df, columns=['product_category'], drop_first = True)


# Define the features and target
X = df.drop(['purchase_date', 'is_returned'], axis = 1)
y = df['is_returned']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Train the randomforest classifier
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)

# Streamlit UI
st.title('Customer Purchase Analysis Dashboard')

# Display the dataset
st.header("Dataset")
st.write(df.head())

# Visualizations
st.header("Visualizations")

# Histogram of the purchase amount
st.subheader("Distribution of Purchase Amount")
fig, ax = plt.subplots()
sns.histplot(df['purchase_amount'], bins = 30, ax = ax)
st.pyplot(fig)

# # Boxplot of purchase_amount by product_category
# st.subheader('Purchase Amount by Product Category')
# fig, ax = plt.subplots()
# sns.boxplot(x = 'product_category', y = 'purchase_amount', data = df, ax=ax)
# st.pyplot(fig)

# Scatterplot of income vs purchase amount
st.subheader('Income vs. purchase amount')
fig, ax = plt.subplots()
sns.scatterplot(x = 'income', y = 'purchase_amount', data = df, ax=ax)
st.pyplot(fig)

# Model Evaluations
st.header('Model Evaluation')
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, y_pred))
st.write('Classification Report:')
st.write(classification_report(y_test, y_pred))

# Predict returning customers
st.header("Predict Returning Customers")
age = st.number_input('Age', min_value=18, max_value=70, value=43)
income = st.number_input('Income', min_value=0, max_value=1000000, value = 60000)
purchase_amount = st.number_input('Purchase Amount', min_value=0, max_value=5000, value = 300)
purchase_year = st.number_input("Purchase Year", min_value= 2019, max_value=2024, value=2022)
purchase_month = st.number_input('Purchase Month', min_value=1, max_value=12, value = 1)

product_categories = df.columns[df.columns.str.startswith('product_category_')]
product_category_values = [0] * len(product_categories)
for i, category in enumerate(product_categories):
    product_category_values[i] = st.number_input(category, min_value=0, max_value=1, value=0)

input_data = [age, income, purchase_amount, purchase_year, purchase_month] + product_category_values
if st.button("Predict"):
    prediction = model.predict([input_data])
    if prediction[0] == 1:
        st.write('Customer is likely to return')
    else:
        st.write('The Customer is not likely to return')