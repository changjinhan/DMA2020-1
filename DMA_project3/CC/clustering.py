from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import v_measure_score
from sklearn.cluster import KMeans
from nltk.corpus import stopwords


categories = ['world', 'us', 'business', 'technology', 'health', 'sports', 'science', 'entertainment']

data = load_files(container_path='text_all', categories=categories, shuffle=True,
                    encoding='utf-8', decode_error='replace',random_state=0)

# TODO - Data preprocessing and clustering

stop_words = set(stopwords.words('english'))

vect=CountVectorizer(stop_words=stop_words,min_df=20, max_df=480)
data_counts=vect.fit_transform(data.data)

data_tf=TfidfTransformer().fit_transform(data_counts)

clst = KMeans(n_clusters=8, init='random', max_iter=1000, n_init=5,random_state=0)
clst.fit(data_tf)


print(v_measure_score(data.target, clst.labels_))



