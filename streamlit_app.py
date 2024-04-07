import streamlit as st
import folium
import pandas as pd
from streamlit_folium import folium_static
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import pickle

@st.cache
def load_tfidf():
    auths = scipy.sparse.load_npz("transformed_authors.npz")
    titles = scipy.sparse.load_npz("transformed_titles.npz")
    
    fileObj = open('auths_trainer.pickle', 'rb')
    auth_train = pickle.load(fileObj)
    fileObj.close()

    fileObj = open('titles_trainer.pickle', 'rb')
    title_train = pickle.load(fileObj)
    fileObj.close()
    return auths, titles, auth_train, title_train


def find_function(path,indexed_journeys_df):
    row = indexed_journeys_df.loc[path]
    return row

def plot_map(my_row):
    start_loc = (my_row['latitude'][0],my_row['longitude'][0])
    m = folium.Map(location=start_loc, zoom_start=5)
    for i, val in enumerate(my_row['country']):
        country = val
        location = (my_row['latitude'][i], my_row['longitude'][i])
        popup = f"{my_row['title'][i]} {val} {my_row['year'][i]}"
        folium.Marker(location=location, popup=popup, icon=folium.Icon(color='blue')).add_to(m)
        if i < len(my_row['country'])-1:
            location_next = (my_row['latitude'][i+1], my_row['longitude'][i+1])
            folium.PolyLine(locations=[location, location_next], color='green', weight=3).add_to(m)
    folium_static(m)

def advanced_find(name,title,auths_db, titles_db, auth_train,title_train,indexed_journeys_df):
    name_vector = auth_train.transform([name])
    ##title_vector = title_train.transform([title])
    #power_search = st.text_input("Search Papers by Title")
    #power_vector = trainer.transform([power_search])
    # word_list = power_search.split(" ") 
    # word_vector = trainer.transform(word_list) 
    # st.write(word_vector)
    # find the cosine similarity between the power search and the first 200 papers
    
    cos_sim = cosine_similarity(name_vector, auths_db)
    # find the index of the paper with the highest cosine similarity
    matching_index = cos_sim.argmax()
    # find the indexis of the top 10 papers with the highest cosine similarity
    top_ten = cos_sim.argsort()[0][-10:]
    top_ten = top_ten[::-1]
    top_sim = cos_sim[0][top_ten]
    # find the title of the paper with the highest cosine similarity
    matching_data = indexed_journeys_df.iloc[top_ten,:]
    path = indexed_journeys_df.index[matching_index]
    # change the index of top_ten_titles to the similarity scores
    matching_data.index = top_sim
    matching_data.index.name = "Similarity Score"
    select_list = matching_data.loc[:,['author_x','title']]
    for i, val in select_list.iterrows():
        select_list.loc[i,'author_x'] = select_list.loc[i,'author_x'][0]
        select_list.loc[i,'title'] = select_list.loc[i,'title'][0]
    st.write(select_list)
    st.write(matching_data)
    st.selectbox("Select An Author", select_list)
    return path

def main():
    alt_df = pd.read_hdf('Path_By_Researchers_With_Year.h5')
    journeys_df = pd.read_hdf('author_journeys.h5')
    journeys_df = alt_df
    indexed_journeys_df = journeys_df.set_index('@path', inplace=False)

    auths_db, titles_db, auth_train, title_train = load_tfidf()
    
    st.title('Researcher Migration')
    path = '/0000-0003-4998-7259'
    path = st.text_input("Write Path Here",path)
    on = st.toggle('Advanced Search')
    if on:
        name = st.text_input("Input researcher name")
        title = st.text_input("Input a paper title to assist the search")
        path = advanced_find(name, title, auths_db, titles_db, auth_train, title_train, indexed_journeys_df)
    row = find_function(path,indexed_journeys_df)
    plot_map(row)

if __name__ == "__main__":
    main()
