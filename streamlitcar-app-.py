import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
model = pickle.load(open("Linear.pkl", "rb"))

def main():
    st.title("Car Selling Price Prediction")
    st.markdown("---")

    st.sidebar.title("Enter Car Details")

    name = st.sidebar.text_input("Car Name")
    year = st.sidebar.number_input("Year", min_value=1950, max_value=2023)
    km_driven = st.sidebar.number_input("Kilometers Driven")
    fuel = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.sidebar.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
    owner = st.sidebar.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    user_data = {
        'name': name,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner
    }

    user_df = pd.DataFrame(user_data, index=[0])

    predict_button = st.sidebar.button("Predict")
    if predict_button:
        prediction = model.predict(user_df)
        st.markdown("---")
        st.subheader("Predicted Selling Price")
        st.success(f"The predicted selling price for the car is: {prediction[0]:.2f} USD")

    # Header and description
    st.header("Welcome to the Car Selling Price Prediction App")
    st.markdown("""
        This application predicts the selling price of a car based on various factors.
        Please enter the car details in the sidebar and click on the **Predict** button.
    """)

if __name__ == "__main__":
    main()
