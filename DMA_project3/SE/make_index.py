import os.path
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

schema = Schema(docID=NUMERIC(stored=True), contents=TEXT)
index_dir = "index"

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

ix = create_in(index_dir, schema)

writer = ix.writer()
stemmizer = LancasterStemmer()
stopWords = set(stopwords.words('english'))

with open('doc/document.txt', 'r') as f:
    text = f.read()
    docs = text.split('   /\n')[:-1]
    for doc in docs:
        br = doc.find('\n')
        docID = int(doc[:br])
        doc_text = doc[br + 1:]

        table = str.maketrans('\n?.,!', '     ')
        doc_text_nomark = doc_text.translate(table)

        new_doc_text = ''
        for word in doc_text_nomark.split(' '):
            if word.lower() not in stopWords:
                word_stem = stemmizer.stem(word.lower())
                new_doc_text += word_stem + ' '

        writer.add_document(docID=docID, contents=new_doc_text)

writer.commit()