import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('finalized_pipe_model.joblib')

# Address,Area,Parking,Elevator,Warehouse
Columns = [ "Area", "Room", "Parking", "Warehouse", 'Elevator',"Address"]
df = pd.read_csv('housePrice.csv')
address = df.Address.unique()
Address = st.selectbox("Address", address)
Area = st.number_input("Area",30,20000000000)
Room = st.number_input("Number of rooms", 0,5)
Parking = st.radio("Parking", ["Yes","No"]) == "Yes"
Elevator = st.radio("Elevator", ["Yes","No"]) == "Yes"
Warehouse = st.radio("Warehouse", ["Yes","No"]) == "Yes"

##-------------- 1st method for creating "output" array------------------------------
#final_vec = [Area, Room, Parking, Warehouse, Elevator, Address]
#output = {i: v for i, v in zip(Columns, final_vec)}

##-------------- 2nd method for creating "output" array------------------------------
#output = {'Area':[Area], 'Room':[Room], 'Parking':[Parking], 'Warehouse':[Warehouse], 'Elevator':[Elevator], 'Address':Address}

##-------------- 3rd method for creating "output" array------------------------------
#output = {'Area': 100000, 'Room': 3, 'Parking': 1, 'Warehouse': 1, 'Elevator': 0, 'Address': ['Darakeh']}

##-------------- 4th method for creating "output" array------------------------------
#output = {'Area':[Area], 'Room':[Room], 'Parking':[Parking], 'Warehouse':[Warehouse], 'Elevator':[Elevator]}
#output['Address'] = pd.Series(Address)

##-------------- 5th method for creating "output" array------------------------------
output = {'Area':[Area], 'Room':[Room], 'Parking':[Parking],'Warehouse':[Warehouse], 'Elevator':[Elevator], 'Address':[Address]}

##############################################################################################################
#############################     3 methods for creating the "dataframe"    ################

X = pd.DataFrame(output)
#X = pd.DataFrame(output,index = [0])
#X = pd.DataFrame({'Area': 100000, 'Room': 3, 'Parking': 1, 'Warehouse': 1, 'Elevator': 0, 'Address':Address},index = [0,1,2])

predictionX = model.predict(X)

if st.button('Predict'):
    st.success(f'The price of this house is {predictionX}!') 
   

