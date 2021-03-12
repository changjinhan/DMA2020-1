import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
import CustomScoring as scoring
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer


def getSearchEngineResult(query_dict):
    result_dict = {}
    ix = index.open_dir("index")

    # with ix.searcher(weighting=scoring.BM25F()) as searcher:
    with ix.searcher(weighting=scoring.ScoringFunction()) as searcher:
        # TODO - Define your own query parser
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0))
        stemmizer = LancasterStemmer()
        stopWords = set(stopwords.words('english'))

        # print(stopWords)
        for qid, q in query_dict.items():

            table = str.maketrans('\n?.,!', '     ')
            q_nomark = q.translate(table)

            new_q = ''
            for word in q_nomark.split(' '):
                if word.lower() not in stopWords:
                    word_stem = stemmizer.stem(word.lower())
                    new_q += word_stem + ' '
            # print(new_q)
            query = parser.parse(new_q.lower())
            results = searcher.search(query, limit=None)
            # for result in results:
            #     print(result.fields()['docID'], result.score)

            result_dict[qid] = [result.fields()['docID'] for result in results]
    return result_dict