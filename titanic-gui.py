import streamlit as st
import pandas as pd
import pickle

data = pd.read_csv("titanic_test.csv")
with open(
    "C:/Users/sambi/Documents/BSAN 6198 - ML Pipeline/Titanic Streamlit GUI/titanic_model.pkl",
    "rb",
) as file:
    model = pickle.load(file)

"""
# Would YOU survive the Titanic??

#### Find out below by putting in your information and hitting the Submit button!
"""

# Step 1: Choose the PClass you belong to
pclass_choice = st.selectbox(
    "Which P-Class are you?", ["First Class", "Second Class", "Third Class"]
)
pclass_map = {"First Class": 1, "Second Class": 2, "Third Class": 3}
pclass = pclass_map[pclass_choice]

# Step 2: Choose your sex
sex = st.radio("What is your sex?", ["Male", "Female"])
sex = sex.lower()

# Step 3: Choose your age
age = st.slider(
    "How old are you?",
    min_value=data["Age"].min(),
    max_value=data["Age"].max(),
    value=30.0,
)

# Step 4: How many siblings or spouses are with you?
sibsp = st.slider(
    "How many Siblings/Spouses are with you?",
    min_value=data["SibSp"].min(),
    max_value=data["SibSp"].max(),
    value=0,
)

# Step 5: Choose the number of parents and children you have
parch = st.slider(
    "How many parents/children are with you?",
    min_value=data.Parch.min(),
    max_value=data.Parch.max(),
    value=0,
)

# Step 6: How much did you pay for your fare?
fare = st.slider(
    "How much was your fare?",
    min_value=data.Fare.min(),
    max_value=data.Fare.max(),
    value=0.0,
)

# Step 7: Choose the place you embarked from
embarked_choice = st.selectbox(
    "Which port did you embark from?", ["Cherbourg", "Queenstown", "Southampton"]
)
embarked_map = {"Cherbourg": "C", "Queenstown": "Q", "Southampton": "S"}
embarked = embarked_map[embarked_choice]

if st.button("Submit"):
    input_data = pd.DataFrame(
        {
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked],
        }
    )
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Prediction: You would have definitely survived the Titanic!")
    elif prediction[0] == 0:
        st.warning(
            "Prediction: Unfortunately, you would have met the same fate as Jack in the icy waters of the North Pole :("
        )
    else:
        st.error(
            "Prediction: Wow... you really stumped the model! It did not return either 1 or 0, so we have no idea if you would survive or not :O"
        )
