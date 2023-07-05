import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
with open('Linear.pkl', 'rb') as file:
    model = pickle.load(file)

  # Header and description
    st.header("Welcome to the Car Selling Price Prediction App")
    st.markdown("""
        This application predicts the selling price of a car based on various factors.
        Please enter the car details in the sidebar and click on the **Predict** button.
    """)

def main():
    st.title("Car Selling Price Prediction")

    # Sidebar
    st.sidebar.header("Enter Car Details")
    name = st.sidebar.text_input("Car Name", value="")
    year = st.sidebar.number_input("Year", min_value=1950, max_value=2023, value=2000)
    km_driven = st.sidebar.number_input("Kilometers Driven", value=50000)
    fuel = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.sidebar.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
    owner = st.sidebar.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    # Predict button
    if st.sidebar.button("Predict"):
        user_data = {'name': name,
                     'year': year,
                     'km_driven': km_driven,
                     'fuel': fuel,
                     'seller_type': seller_type,
                     'transmission': transmission,
                     'owner': owner}

        user_df = pd.DataFrame(user_data, index=[0])
        prediction = model.predict(user_df)

        st.subheader("Predicted Selling Price")
        st.success("The predicted selling price for the car is: {}".format(prediction[0]))

  

if __name__ == "__main__":
    main()
