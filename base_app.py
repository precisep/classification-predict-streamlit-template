"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os


# Data dependencies
import pandas as pd

#NLP libraries
from nltk.corpus import stopwords
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
#Count Vectorizer
team_vectorizer = open("resources/vectorizer.pkl","rb")
sentiments_cv = joblib.load(team_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title('Tweet Classifer')
	st.subheader('Climate change tweet classification')

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	model_options = ['Base_model','Logistic_Regression','Naive_Bayes','K-Neighbors']#,'Artificial_Neural_Network']
	model_selection = st.sidebar.selectbox("Choose Model",model_options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Share Your Tweet","Type Here")

		if st.button("Classify") :
			if model_selection == 'Base_model':
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

			elif model_selection == 'Logistic_Regression':
				# Transforming user input with vectorizer
				vect_text = sentiments_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/lr.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(prediction))
				
			elif model_selection == 'Naive_Bayes':
				# Transforming user input with vectorizer
				vect_text = sentiments_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/nb_clf.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(prediction))

			elif model_selection == 'K-Neighbors':
				# Transforming user input with vectorizer
				vect_text = sentiments_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/knn.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(prediction))

			
			else:
				st.text('Please Select a model you would like to utilize for making predictions')

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

#Although climate change is real, it's been sensationalized by media and political leaders to drive a politics of fear amongst the masses...
