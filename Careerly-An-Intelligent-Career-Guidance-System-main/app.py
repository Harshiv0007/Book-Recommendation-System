import pickle

from flask import Flask, render_template, request
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# import numpy as np
with open("topfifty.pkl", "rb") as f:
    data = pickle.load(f)

with open("sample.pkl", "rb") as f:
    books = pickle.load(f)

with open("new_top_authors.pkl", "rb") as f:
    top_authors = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html", book_name=list(data['title'].values),
                           author=list(data['author'].values),
                           image=list(data['coverImg'].values),
                           rating=list(data['likedPercent'].values))


@app.route('/recommend')
def recommend_ui():
    return render_template("recommend.html")


@app.route('/recommend_books', methods=['post'])
def recommend():
    try:

        user_input = request.form.get('user_input')
        header = f"Recommendations based on the provided book: '{user_input}'"
        print(header)

        features = ['title', 'author', 'genres']
        books['text'] = books[features].apply(lambda x: ' '.join(x[:2]), axis=1)
        vectorizer = TfidfVectorizer()
        vectors_title_author = vectorizer.fit_transform(books['text'])
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(books['genres'])

        vectors_combined = sparse.hstack((vectors_title_author, genres_encoded))
        similarity_matrix = cosine_similarity(vectors_combined)

        books.drop('text', axis=1, inplace=True)

        target_node = user_input
        # print(target_node)

        target_index = books[books['title'].str.contains(target_node, case=False, na=False)].index[0]

        similar_books_indices = similarity_matrix[target_index].argsort()[::-1][1:]

        table_data = []
        for i, index in enumerate(similar_books_indices[:10], start=1):
            item = []
            book_title = books.loc[index, 'title']
            book_author = books.loc[index, 'author']
            book_image = books.loc[index, 'coverImg']
            similarity_percentage = similarity_matrix[target_index][index] * 100
            item.extend([i, book_title, book_author, book_image, f"{similarity_percentage:.2f}%"])
            table_data.append(item)
        print(table_data)
        # print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

        return render_template("recommend.html", data=table_data, user_input=user_input)
    except Exception as e:
        return render_template("recommend.html")


@app.route('/category')
def category_ui():
    return render_template("category.html")


@app.route('/recommend_category', methods=['post'])
def recommend_category():
    user_input = request.form.get('user_input')
    header = f"Recommendations based on the provided book: '{user_input}'"

    genre_books = books[
        books['genres'].str.split(',').str[0].str.strip().str.contains(user_input, case=False, na=False)]
    test = books['genres'].str.split(',').str[0:2].str.strip()
    print(test)

    # Sort books by a relevant criteria (e.g., rating)
    sorted_genre_books = genre_books.sort_values(by='rating', ascending=False)

    # Extract book titles
    recommended_books = sorted_genre_books[['title', 'author', 'coverImg']]
    # print(list(recommended_books['coverImg'].values))

    return render_template("category.html", user_input=user_input, image=list(recommended_books['coverImg'].values),
                           book_name=list(recommended_books['title'].values),
                           author=list(recommended_books['author'].values))


@app.route('/authors')
def authors_ui():
    return render_template("authors.html",
                           author=list(top_authors['author'].values),
                           numBooks=list(top_authors['numBooks'].values),
                           likedPercent=list(top_authors['likedPercent'].values),
                           coverImg=list(top_authors['coverImg'].values))


if __name__ == '__main__':
    app.run(debug=True)
