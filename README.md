# Title: Categorization-and-search-engine-for-Covid-19-research-studies

## Authors: Srikanth Babu Mandru

## Summary:

During this Covid-19 pandemic, there is a urgent need of proper medication to eradicate the disease from our planet. Health care professionals and researchers around the world working continuously in order to discover the vaccine. In this process, the researchers have been publishing their ideas and ways to deal with this kind of virus. It is difficult to keep up with these numerous articles being published daily. This project mainly focus on solving the problem of identifying the groups (categorize) of articles and the important topics or keywords for each those groups, which helps us to categorize the articles and focus more on the those that are of high interest. Further, this project also aims at building a web application to provide better interactivity with the models built in this project.

## About data:

For this project, dataset has been taken from 'Kaggle' ![3](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). This dataset was created by the Allen Institute for AI in partnership with the Chan Zuckerberg Initiative, Georgetown University’s Center for Security and Emerging Technology, Microsoft Research, IBM, and the National Library of Medicine - National Institutes of Health, in coordination with The White House Office of Science and Technology Policy.

## Purpose of this project: 

   - Companies, Eductional institutions and many organizations are being flooded with large corpora of text daily and they need data science tools to draw valuable insights from the data and take key decisions to develop business and inturn improve the customer satisfaction towards the company's product. Often, it is case that we need to know the underlying groups or structures in our data, and possibly, get more information about each of them in form of short summaries or make a search on data with really important keywords. This project aims at dealing with similar kind of problem except that we are doing this project with covid-19 research papers.

  - Often, the raw data comes in an unexpected and messy way. We need to tidy up the data and transform it to the form that is useful for the analysis and modelling to make inferences. Mostly, during analysis, we need to find the hidden structure to find the patterns/groups in the data. Finding the patterns gives us more intuition towards solving the problem and reinforces our decisions. Since, the data doesn't have labels, we have to stick with the unsupervised techniques to deal study about the data. This project provides a somewhat comprehensive view on most popular/widely used unsupervised methods and the advantages or drawbacks of each method. The same tools work for other problems as well whether it's structured or unstructured data.

## Approach: 
  
  As the data comes in the form of text, initially, the text will be parsed and convert each document's text into machine understandable way for machine learning tasks using 'Term Frequency–inverse Document Frequency (TF-IDF)' vectorization method. For categorizing the research papers, we use the clustering techniques from unsupervised machine learning paradigm and group the research articles together so that the close articles stay in same group and different types of articles will be far away from each other. Since, the text data is huge and sparse in nature, dimensionality reduction technique, namely PCA will be applied, before the Clustering and other modelling, to shrink the the document representation and remove the unneccesary features from our data. For visualization of the articles, it is difficult to plot with higher dimensional data. Thus, t-Stochastic Neighbour Embedding (t-SNE), a the dimensionality reduction technique for visualization of high dimensional data points, will be applied in order to represent the groups (labelled clusters) visually.
  To find the important words in each article, topic modelling is performed on each of the groups seperately using the 'LatentDirchiletAllocation' (LDA) method and the quality of each model is measured through the perplexity score. For the part of web application, I will be using either the Bokeh or Flask or JS.


## Methods: 


## Results:


## References: 

1. http://www.mmds.org/

2. https://scikit-learn.org/stable/

3. Kaggle : https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
