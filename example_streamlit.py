# Import the packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#Title of application
st.title("Streamlit Dashboard")

# Subtitle of my application
st.subheader("An application to demonstrate proficiency in Streamlimt")

# Upload data to visualize
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    #read the file to a dataframe
    data = pd.read_csv(uploaded_file)

    #Display the dataframe
    st.write("Dataframe")
    st.write(data)

    # Line Chart
    st.line_chart(data)

    # Give additional visualization using matplotlib
    st.subheader("Matplotlib Visualization")
    fig, ax = plt.subplots()
    data.plot(ax = ax)
    st.pyplot(fig)

else:
    st.write("Please upload a csv file to procee.d")