import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import os
monkey = RandomForestRegressor()
x = pd.DataFrame({
     "Hours":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
     "drink":["coffee","juice","tea","coffee","juice","tea","coffee","coffee","juice","tea","coffee","juice","tea","coffee"]
})

le = LabelEncoder()
x['drink'] = le.fit_transform(x['drink'])
y = np.array([44,60,65,70,80,91,94,100,122,140,170,180,191,194])
pca = PCA(n_components=1)
x = pca.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)
monkey.fit(x_train,y_train)
st.title("توقع درجاتك مع الذكاء الاصطناعي الخاص بنا ")
y_pred_test = monkey.predict(x_test)
accuracy = monkey.score(x_test,y_test) * 100
print(f"دقة النموذج الحاليا {accuracy}")
user_saat = st.number_input(" :كم ساعة ذاكرت ")
user_drink = st.selectbox("اختار مشروبك:", options=[0, 1, 2], format_func=lambda x: ["قهوة", "عصير", "شاهي"][x])
user_input = [[user_saat,user_drink]]
input_pca = pca.transform(user_input)
opo = monkey.predict(input_pca)
st.success(f"توقع مونكي وهو{opo}")
