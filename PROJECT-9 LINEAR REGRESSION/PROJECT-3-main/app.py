
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Insurance Prediction", page_icon="🛡️")
st.title("🛡️ Life Insurance Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("insurance_data.csv")

df=load_data()
X=df[["age"]]
y=df["bought_insurance"]
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
acc=model.score(X_test,y_test)
st.write(f"Model Accuracy: **{acc:.2%}**")
age=st.slider("Select Age",1,100,30)
if st.button("Predict"):
    pred=model.predict([[age]])[0]
    prob=model.predict_proba([[age]])[0][1]
    if pred:
        st.success(f"Likely to buy insurance ({prob:.1%})")
    else:
        st.error(f"Unlikely to buy insurance ({prob:.1%})")
with st.expander("Dataset"):
    st.dataframe(df)
