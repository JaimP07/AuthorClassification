import pandas as pd
import numpy as np
import gensim
from gensim.parsing.preprocessing import preprocess_documents
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot as plt
from scipy import sparse

#Set the number of rows of blog data to be used
nr = 1000

#Reads in the data
df = pd.read_csv('C:/Users/pache/Documents/CS159Labs/archive/blogtext.csv',sep=',',usecols = ['id','topic','text'],nrows = nr)
tc = df['text'].values

pc = preprocess_documents(tc)
diction = gensim.corpora.Dictionary(pc)

#'id' represents authors, but can be set to 'topic' as well
feature = 'id'

#Contains the vectors representing the words for each text:
bowc = [diction.doc2bow(t) for t in pc]

#Construct a matrix of vectors for each blog
bowc_mat = np.zeros((len(bowc),len(diction)))
for i in range(len(bowc)):
    for j in bowc[i]:
        bowc_mat[i,j[0]] = j[1]


#Balances out word weight
tfidf = gensim.models.TfidfModel(bowc, smartirs='npu')

#Cosine similarity matrix for the vectors
index = gensim.similarities.MatrixSimilarity(tfidf[bowc], dtype = 'float16')

u1 = 0
u2 = nr

#Determine average accuracy for all blogs
avg = 0
for q in list(range(u1,u2)):

    #Determine how many blogs from the current blog author exist in our data set
    c = len(df[df[feature]==df[feature].iloc[q]])

    #Get vector of current blog
    vec_bow_tfidf = tfidf[bowc[q]]
    
    #Extracts similarities:
    sims = index[vec_bow_tfidf]
    
    #Iterate through the best similarity scores, and compute accuracy
    cur = 0
    for s in sorted(enumerate(sims), key=lambda item: -item[1])[:c]:
        if (df[feature].iloc[s[0]] == df[feature].iloc[q]):
            cur += 1
    avg += cur/c
print(avg/(u2-u1))


#Reduces the data to 2 dimensions for plotting
pca = PCA(n_components=2)
pc = pca.fit_transform(bowc_mat)

#Builds dictionary that assigns each author a unique color
colordict = dict.fromkeys(df[feature].tolist())
L = colordict.keys()
amt = 1/len(L)
cur = amt
for i in L:
    colordict[i] = cur
    cur = cur + amt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)

for i in range(len(pc)):
    v = colordict[df[feature].iloc[i]]
    ax.scatter(pc[i][0],pc[i][1], color = (0.2,v,0.5), s=5)
plt.show()
