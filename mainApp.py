import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('finalized_pipe_model.joblib')
#st.title ("Artificial Estate Agent")
#st.header("House Price Prediction 📊")
#st.image("https://images.adsttc.com/media/images/6321/62f5/34b7/7601/6e91/d3fe/medium_jpg/ferdows-house-naghshekhak-architectural-group_1.jpg?1663132473")

# Address,Area,Parking,Elevator,Warehouse
Columns = [ "Area", "Room", "Parking", "Warehouse", 'Elevator',"Address"]
df = pd.read_csv('housePrice.csv')
address = df.Address.unique()
Address = st.multiselect("Address", address) 
#Address = ["Darekeh"]
Area = st.number_input("Area",30,20000000000)
Room = st.number_input("Number of rooms", 0,5)
Parking = st.radio("Parking", ["Yes","No"]) == "Yes"
Elevator = st.radio("Elevator", ["Yes","No"]) == "Yes"
Warehouse = st.radio("Warehouse", ["Yes","No"]) == "Yes"

final_vec = [Area, Room, Parking, Warehouse, Elevator, Address]
#output = {i: v for i, v in zip(Columns, final_vec)}
output = {'Area':[Area], 'Room':[Room], 'Parking':[Parking], 'Warehouse':[Warehouse], 'Elevator':[Elevator], 'Address':Address}
#output = {'Area': 100000, 'Room': 3, 'Parking': 1, 'Warehouse': 1, 'Elevator': 0, 'Address': ['Darakeh']}
#X = pd.DataFrame({'Area': 100000, 'Room': 3, 'Parking': 1, 'Warehouse': 1, 'Elevator': 0, 'Address':Address},index = [0,1,2])
#output = {'Area':[Area], 'Room':[Room], 'Parking':[Parking], 'Warehouse':[Warehouse], 'Elevator':[Elevator]}
#output['Address'] = pd.Series(Address)
X = pd.DataFrame(output)
#X = pd.DataFrame(output,index = [0])
#predictionX = model.predict(X)

predictionX = X
#row = np.array(final_vec) 

if st.button('Predict'):
    st.success(f'The price of this house is {predictionX}!') 
   

