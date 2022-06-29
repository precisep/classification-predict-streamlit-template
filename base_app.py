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
import matplotlib.pyplot as plt
import seaborn as sns

#image dependencies
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
#Count Vectorizer
team_vectorizer = open("resources/vectorizer.pkl","rb")
sentiments_cv = joblib.load(team_vectorizer)

#seniments dictionary has
sentiments_dict = { -1 : 'Anti Climate Change Sentiment',
					0  : 'Neutral Climate Change Sentiment',
					1  : 'Pro Climate Change Sentiment',
					2  : 'News Climate Change Sentiment'
}

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """


	
	image = Image.open('./resources/imgs/company_logo.png')

	st.sidebar.image(image, caption='Green Tech Solutions (Pty) Ltd', width=300)

	

	# Creating sidebar with selection box 
	# you can create multiple pages this way
	#page_option = ['About', 'Exploratory Data Analysis (EDA)', 'Home']
	#page_selection = st.sidebar.selection('Pages',page_option)
	options = ["Prediction", "Information","About","Exploratory Data Analysis (EDA)"]
	selection = st.sidebar.selectbox("Choose Option", options)

	

	# Building out the "Information" page
	if selection == "Information":
		st.markdown("<h1 style='text-align: center;'>Tweet Classifer</h1>", unsafe_allow_html=True)
		st.subheader('Climate change tweet classification')
		st.info("General Information")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		# Creates a main title and subheader on your page :
		# these are static across all pages
		st.markdown("<h1 style='text-align: center;'>Tweet Classifer</h1>", unsafe_allow_html=True)
		st.subheader('Climate change tweet classification')
		st.info("Prediction with Machine Learning Models")

		# Creating a text box for user input
		tweet_text = st.text_area("Share Your Tweet","Type Here")

		#List of models to choose from
		model_options =['Base_model','Logistic_Regression','Naive_Bayes','K-Neighbors']#,'Artificial_Neural_Network'
		model_selection = st.sidebar.selectbox('Available Models',model_options)
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
				st.success("Text Categorized as: {}".format(sentiments_dict[prediction[0]]))

			elif model_selection == 'Logistic_Regression':
				# Transforming user input with vectorizer
				vect_text = sentiments_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/lr.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(sentiments_dict[prediction[0]]))
				
			elif model_selection == 'Naive_Bayes':
				# Transforming user input with vectorizer
				vect_text = sentiments_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/nb_clf.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(sentiments_dict[prediction[0]]))

			elif model_selection == 'K-Neighbors':
				# Transforming user input with vectorizer
				vect_text = sentiments_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/knn.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(sentiments_dict[prediction[0]]))

			
			else:
				st.text('Please Select a single model you would like to utilize for making predictions')
			# Meaning of the classes
			st.subheader("Meaning Of The Classes")
			st.markdown('News: the tweet links to factual news about climate change')
			st.markdown('Pro: the tweet supports the belief of man-made climate change')
			st.markdown('Neutral: the tweet neither supports nor refutes the belief of man-made climate change')
			st.markdown('Anti: the tweet does not believe in man-made climate change')

	if selection == 'About':
		st.title('Green Tech Solutions (Pty) Ltd')
		st.markdown('We are an innovative technological company, established in 2022. We specialize in providing accurate and robust tech solutions\n'
                    'for companies who want to access a broad base of consumer sentiments, spanning multiple demographic and geographic categories.\n'
                    'Our products help companies gain more insights about the impact of their products, which can assist in informing future marketing strategies.\n'
                    'Our undying passion for our planet and eco-friendly products is what drives us to collect valuable information to assist companies in making an informed decision when producing their favorite products. ')

		st.title('Meet Our Team')
		st.markdown('Our team is made up of enthusiastic data scientists who are passionate about data and solving challenging problems.')			
		
		image = Image.open('./resources/imgs/kgotsatso.jpg')
		st.image(image,'CEO: Kgotsatso Malapane,\n'
						'Data Scientist', width=250)	
		image = Image.open('./resources/imgs/atlegang.jpg')
		st.image(image,'Co-Founder: Atlegang Mogane,\n'
						'Data Scientist', width=250)	
		image = Image.open('./resources/imgs/thabang.jpg')
		st.image(image,'CFO: Thabang Mokoena,\n'
						'Data Scientist', width=250)
						
	#exploratory data analysis page.					
	if selection == "Exploratory Data Analysis (EDA)":

		st.markdown("<h1 style='text-align: center;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
		
		#analysing the labels
		fig = plt.figure(figsize=(10, 4))	
		sns.countplot(raw['sentiment'], palette= 'Pastel1',hue = raw['sentiment'])
		plt.legend(title = 'Sentiments', loc = 'upper left', labels= ['Anti','Neutral','Pro','News'])
		plt.title('Climate Change Sentiments', fontweight = 'bold')
		st.pyplot(fig)

		st.markdown('The model was build using survay results presented in the raw data table. Figure Above show the distribution of the various sentiments collected to for model building.\n'
					"It can be observed that the label sentiments are imbalanced, giving me weight to the 'Pro' sentiment as it has more datapoints.")
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

#Although climate change is real, it's been sensationalized by media and political leaders to drive a politics of fear amongst the masses...
