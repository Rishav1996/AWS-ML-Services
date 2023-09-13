import streamlit as st
from transformers import pipeline
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px 
from io import StringIO
import boto3
import glob
from nltk.corpus import stopwords


@st.cache_data
def get_classifier():
   classifier = pipeline("summarization")
   return classifier


@st.cache_data
def get_qa_model():
   qa_model = pipeline("question-answering")
   return qa_model


@st.cache_data
def get_wordcloud(text):
   return WordCloud().generate(text)


def get_sankey_chart(df):
   labels = list(df['Type'].unique())
   labels.extend(df['Text'].unique())
   source = list(map(lambda x:labels.index(x), df['Type']))
   target = list(map(lambda x:labels.index(x), df['Text']))
   fig = go.Figure(data=[go.Sankey(
      node = dict(
         label = labels,
      ),
      link = dict(
         source = source,
         target = target,
         value = df['Score'].map(lambda x:x*100).map(int),
         color='lightsteelblue',
   ))])
   return fig


st.set_page_config('Text Analytics', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

list_of_files = list(map(lambda x:str(x).split('\\')[-1].split('.')[0], glob.glob('./text-analytics-sample/*')))
file_option = st.sidebar.selectbox("Please select a file", options=list_of_files)
st.sidebar.divider()
file_text = st.sidebar.file_uploader('Upload file', type='txt')
actual_text = ""
client = boto3.client('comprehend', region_name='us-east-1')

if file_text != None:
   actual_text = ''.join(StringIO(file_text.getvalue().decode("utf-8")))

if file_option != None:
   print(file_option)
   for i in glob.glob('./text-analytics-sample/*'):
      if i.find(file_option) > 0:
         file_text = i
         break
   with open(file_text, 'r') as f:
      actual_text = f.readline()
      actual_text = ''.join(actual_text)

if actual_text != "":

   @st.cache_data
   def text_summarization(text):
      return get_classifier()(text)[0]['summary_text']

   @st.cache_data
   def question_answer(question):
      return get_qa_model()(question = question, context = actual_text)['answer']

   tab1, tab2, tab3 = st.tabs(["Text Analysis", "Most Frequent Words", "Sentiment Analysis"])

   with tab1:

      st.header('Text Summarization')
      st.markdown(text_summarization(actual_text))

      st.header('Question Answer')
      col1, _ = st.columns(2)
      with col1:
         question = st.text_input('Enter your question')
         if question != '':
            st.text(f"Answer : {question_answer(question)}")
   
   with tab2:
      
      wordcloud = get_wordcloud(actual_text)

      fig = plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      plt.show()
      st.pyplot()

      most_freq = [[i, str(actual_text).lower().count(i)] for i in str(actual_text).lower().split(' ') if i not in stopwords.words('english')]
      most_freq = [[i, j] for i,j in most_freq if str(i).isalpha() and len(i) > 3]
      most_freq = pd.DataFrame(most_freq, columns=['Word', 'Count'])
      most_freq.drop_duplicates(inplace=True)
      most_freq = most_freq.sort_values('Count', ascending=False).head(10)
      fig=px.bar(most_freq, x='Count', y='Word', orientation='h', title="Most Frequent Words")
      st.write(fig)
      
   
   with tab3:
      response = client.detect_sentiment(Text=actual_text, LanguageCode='en')
      sentiment_label = response['Sentiment']
      sentiment_score = response['SentimentScore']
      st.metric('Overall Sentiment', sentiment_label)
      col_t3 = st.columns(4)

      for c, k in zip(col_t3, sentiment_score.keys()):
         with c:
            st.metric(k, round(sentiment_score[k] * 100, 2))

      response = client.detect_entities(Text=actual_text, LanguageCode='en')
      entity_res = pd.DataFrame(response['Entities'])
      entity_res = entity_res[['Score', 'Type', 'Text']]
      entity_res.sort_values(['Type', 'Text'], inplace=True)
      entity_res.drop_duplicates(inplace=True)
      fig = get_sankey_chart(entity_res)
      st.markdown('Entity Mapping Visuals')
      radio_options = st.radio(label='Select',options=['Chart', 'Table'], index=0, horizontal=True)
      if radio_options == 'Chart':
         st.plotly_chart(fig)
      else:
         entity_res.index = entity_res['Text']
         entity_res.drop(columns=['Text', 'Score'], inplace=True)
         st.dataframe(entity_res, 500)

