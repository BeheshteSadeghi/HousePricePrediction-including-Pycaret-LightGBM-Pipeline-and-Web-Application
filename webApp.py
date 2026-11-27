import pandas as pd
import streamlit as st

from pricing import FEATURES, fit_estimator, load_data


@st.cache_resource
def trained_estimator():
    frame = load_data()
    return fit_estimator(frame[FEATURES], frame['Price']), sorted(frame['Address'].unique())


def main():
    st.set_page_config(page_title='Tehran Listing Price Estimator', layout='centered')
    st.title('Tehran listing price estimator')
    st.write('Explore estimated asking prices from the repository’s historical listing sample.')
    model, neighborhoods = trained_estimator()
    with st.form('listing'):
        address = st.selectbox('Neighborhood', neighborhoods)
        left, right = st.columns(2)
        area = left.number_input('Area (m²)', 10, 2000, 80)
        rooms = right.number_input('Rooms', 0, 10, 2)
        parking = st.checkbox('Parking')
        warehouse = st.checkbox('Storage')
        elevator = st.checkbox('Elevator')
        submitted = st.form_submit_button('Estimate price')
    if submitted:
        listing = pd.DataFrame([[area, rooms, parking, warehouse, elevator, address]], columns=FEATURES)
        st.metric('Estimated asking price', f'{model.predict(listing)[0]:,.0f}')
        st.caption('The estimate uses the original units of the dataset’s Price column.')


if __name__ == '__main__':
    main()
