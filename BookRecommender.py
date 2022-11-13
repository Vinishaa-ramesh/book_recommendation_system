import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.metrics.distance import edit_distance
from nltk.util import ngrams

books=pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding='latin-1')
books=books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]# feature engineering book dataset
books.rename(columns={'Book-Title':'title','Book-Author':'author','Year-Of-Publication':'year','Publisher':'publisher'},inplace=True)
users=pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding='latin-1')
users.rename(columns={'User-ID':'user_id','Location':'location','Age':'age'},inplace=True)
ratings=pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding='latin-1')
ratings.rename(columns={'User-ID':'user_id','Book-Rating':'rating'},inplace=True)
x=ratings['user_id'].value_counts()>200
y=x[x].index
ratings=ratings[ratings['user_id'].isin(y)]
ratings_with_books=ratings.merge(books, on='ISBN')
number_rating=ratings_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns={'rating':'number_of_ratings'},inplace=True)
final_rating=ratings_with_books.merge(number_rating, on='title')
final_rating=final_rating[final_rating['number_of_ratings']>=50]
final_rating.drop_duplicates(['user_id','title'], inplace=True)
book_pivot=final_rating.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0,inplace=True)
book_sparse=csr_matrix(book_pivot)
model=NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

book_corrected = book_pivot.index
spellings_series = pd.Series(book_corrected)

def recommend_book(book_name):
    book_ind=np.where(book_pivot.index==book_name)
    if len(book_ind[0])!=0:
        book_id=book_ind[0][0]
        distances,suggestions=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
        for i in range(len(suggestions)):
            if i==0:
                st.write('**Recommendations for you:**')
            if not i:
                book_x=book_pivot.index[suggestions[i]]
        for i in range(len(book_x)):
            st.write(book_x[i])
    else:
        outcomes = ""
        distances = ((edit_distance(book_name, word), word)
                            for word in book_corrected)
        closest = min(distances)
        outcomes = closest[1]
        st.write("_did you mean:_",outcomes)
        recommend_book(outcomes)

# recommend_book(input("Enter book: "))
with st.form("my_form"):
    st.title('Book Recommendation System')
    # book_name = st.selectbox(
    #     'Select a book from below or type a title you need to find...',
    #     books['title'][1:25])
    title = st.text_input('Book Name: ',placeholder='Enter book name...')
    submitted = st.form_submit_button("Submit")
    if submitted and title:
        recommend_book(title)
# st.write(recommend_book(book_name))