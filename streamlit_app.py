import streamlit as st
from streamlit import components
import numpy as np
import pandas as pd
import requests

# helper function for blank streamlit lines
def V_SPACE(lines):
    for _ in range(lines):
        st.write('&nbsp;')


st.set_page_config(layout="wide")

####################
### INTRODUCTION ###
####################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Financial News Sentiment Analysis')
with row0_2:
    V_SPACE(1)
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.subheader("Enter financial news text and click score to determine the sentiment")
    V_SPACE(1)
    
#################
### SELECTION ###
#################

with st.form("my_form"):
    fintext = st.text_input('Input text', 'there is a shortage of capital, and we need extra financing')
    scored = st.form_submit_button("Score")

results = list()

response = requests.post("https://prod-field.cs.domino.tech:443/models/641356d19faea51184b824ef/latest/model",
    auth=(
        "z66uvumdWXlLgaYm4flzFCykLseVpKN3EXfWPHrncyPIZ8X8uhvIUDRLGI2A1Wby",
        "z66uvumdWXlLgaYm4flzFCykLseVpKN3EXfWPHrncyPIZ8X8uhvIUDRLGI2A1Wby"
    ),
    json = {
              "data": {
                "sentence": fintext
              }
            }
    )
results.append(response.json().get('result'))

### Results ###

labels = []
scores = []
for s in results[0]:
    labels.append(s["label"])
    scores.append(s["score"])
    
df = pd.DataFrame(columns = ['label', 'score'])
df.label = labels
df.score = scores
df_sorted = df.sort_values(by='score', ascending=False)
result_text = df_sorted.label.values[0]
result_prob = round(df_sorted.score.values[0], 4)
    
#################
### VIZ ###
#################

#import plotly.graph_objects as go
import plotly.express as px

fig = px.bar(df, x='label', y='score',
             hover_data=['label', 'score'], color='score', height=400, 
             color_continuous_scale=px.colors.sequential.Viridis_r)

fig.update_layout(paper_bgcolor = "#0e1117", font = {'color': "white", 'family': "Arial"})
 
row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:    
    st.subheader('The sentiment of this financial text is ' + result_text + ' with probability of ' + str(result_prob))
    V_SPACE(1)
    st.subheader('Results For Each Label')
    st.plotly_chart(fig, use_container_width=True)
