# TODO: CHANGE THIS FILE NAME TO DMA_project2_team##.py
# EX. TEAM 1 --> DMA_project2_team01.py

# TODO: IMPORT LIBRARIES NEEDED FOR PROJECT 2
import mysql.connector
import os
import csv
import surprise
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise.model_selection.search import GridSearchCV
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import tree
import graphviz
from mlxtend.frequent_patterns import association_rules, apriori

np.random.seed(0)

# TODO: CHANGE GRAPHVIZ DIRECTORY
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# os.environ["PATH"] += os.pathsep + '/usr/local/Cellar/graphviz/2.44.0/bin/' # for MacOS

# TODO: CHANGE MYSQL INFORMATION, team number 
HOST = 'localhost'
USER = 'root'
PASSWORD = ''
SCHEMA = 'DMA_team10'
team = 10


# PART 1: Decision tree 
def part1():
    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)

    # TODO: Requirement 1-1. MAKE pro_mentor column
    cursor.execute('ALTER TABLE mentor ADD pro_mentor TINYINT(1) DEFAULT 0')

    promentor = open('pro_mentor_list.txt', 'r')
    p = promentor.readlines()

    for var in range(0, len(p)):
        a = p[var]
        pro = a.replace('\n', '')
        cursor.execute('UPDATE mentor SET pro_mentor=1 WHERE id=\'%s\'' % (pro))

    promentor.close()

    # TODO: Requirement 1-2. WRITE MYSQL QUERY AND EXECUTE. SAVE to .csv file

    fopen = open('DMA_project2_team%02d_part1.csv' % team, 'w', encoding='utf-8')

    cursor.execute('''
        SELECT id, pro_mentor, age, have_introduction,have_field,num_of_answers,avg_of_answer_score,avg_of_answer_body,
        num_of_groups, avg_of_group_members, num_of_emails, num_of_tags
        FROM (SELECT mentor.id AS id, mentor.pro_mentor AS pro_mentor,
            truncate(( unix_timestamp('2020-01-01 00:00:00') - unix_timestamp(mentor.joined_date))/3600,0) AS age,
            if(isnull(mentor.introduction),'0','1') AS have_introduction, if(isnull(mentor.field),'0','1') AS have_field,
            COUNT(answer.mentor_id) AS num_of_answers, AVG(answer.score) AS avg_of_answer_score, AVG(answer.body) AS avg_of_answer_body 
            FROM mentor LEFT JOIN answer ON answer.mentor_id = mentor.id 
            GROUP BY mentor.id) AS A1
        LEFT JOIN
            (SELECT mg.mentor AS id2, COUNT(mg.id) AS num_of_groups 
            FROM mentoring_group AS mg GROUP BY mg.mentor) AS A2
        ON A1.id=A2.id2
        LEFT JOIN (SELECT id AS id3,  AVG(num_of_group_members) AS avg_of_group_members 
            FROM (SELECT mg.mentor AS id, COUNT(mentee_id) AS num_of_group_members 
                    FROM group_membership AS gm RIGHT JOIN mentoring_group AS mg ON gm.group_id=mg.id GROUP BY gm.group_id) AS temp
            GROUP BY id
            ) AS A3
        ON A1.id =A3.id3
        LEFT JOIN (SELECT email.recipient_id, COUNT(*) AS num_of_emails FROM mentor LEFT JOIN email ON mentor.id = email.recipient_id GROUP BY email.recipient_id) AS A4
        ON A1.id = A4.recipient_id
        LEFT JOIN (SELECT tag_mentor.mentor_id, COUNT(*) AS num_of_tags FROM mentor LEFT JOIN tag_mentor ON mentor.id = tag_mentor.mentor_id GROUP BY tag_mentor.mentor_id) AS A5
        ON A1.id = A5.mentor_id;
    ''')

    rows = cursor.fetchall()
    column = [i[0] for i in cursor.description]
    w = csv.writer(fopen, lineterminator='\n')
    w.writerow(column)
    w.writerows(rows)
    fopen.close()

    # -------

    # TODO: Requirement 1-3. MAKE AND SAVE DECISION TREE
    # gini file name: DMA_project2_team##_part1_gini.pdf
    # entropy file name: DMA_project2_team##_part1_entropy.pdf

    data = pd.read_csv("DMA_project2_team10_part1.csv")
    feature_names = ['age', 'have_introduction', 'have_field', 'num_of_answers', 'avg_of_answer_score',
                     'avg_of_answer_body', 'num_of_groups', 'avg_of_group_members', 'num_of_emails', 'num_of_tags']
    label_name = ['pro_mentor']

    data = data.fillna(0)

    df = pd.DataFrame(data)
    x = df[feature_names].values.tolist()
    y = df[label_name].values.tolist()

    DT1 = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=10)
    DT1.fit(x, y)
    DT2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10)
    DT2.fit(x, y)

    graph = tree.export_graphviz(DT1, out_file=None,
                                 feature_names=['age', 'have_introduction', 'have_field', 'num_of_answers',
                                                'avg_of_answer_score',
                                                'avg_of_answer_body', 'num_of_groups', 'avg_of_group_members',
                                                'num_of_emails', 'num_of_tags'], class_names=['normal', 'PRO'])
    graph = graphviz.Source(graph)
    graph.render('DMA_project2_team10_part1_gini', view=True)

    graph = tree.export_graphviz(DT2, out_file=None,
                                 feature_names=['age', 'have_introduction', 'have_field', 'num_of_answers',
                                                'avg_of_answer_score',
                                                'avg_of_answer_body', 'num_of_groups', 'avg_of_group_members',
                                                'num_of_emails', 'num_of_tags'], class_names=['normal', 'PRO'])
    graph = graphviz.Source(graph)
    graph.render('DMA_project2_team10_part1_entropy', view=True)

    # -------

    # TODO: Requirement 1-4. Don't need to append code for 1-4

    # -------

    cursor.close()
    

# PART 2: Association analysis
def part2():
    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)

    # TODO: Requirement 2-1. CREATE VIEW AND SAVE to .csv file
    cursor.execute('''
    CREATE VIEW tag_score
    AS SELECT id AS tag_id, name AS tag_name, num_mentor, num_mentee, num_question, num_mentor+num_mentee+num_question AS score
    FROM tag,
    (
    SELECT tag_id, COUNT(*) AS num_mentor
    FROM tag_mentor
    GROUP BY tag_id
    ) AS mr,
    (
    SELECT tag_id, COUNT(*) AS num_mentee
    FROM tag_mentee
    GROUP BY tag_id
    ) AS me,
    (
    SELECT tag_id, COUNT(*) AS num_question
    FROM tag_question
    GROUP BY tag_id
    ) AS q
    WHERE tag.id = mr.tag_id AND tag.id = me.tag_id AND tag.id = q.tag_id
    ORDER BY score DESC
    LIMIT 50;
    ''')

    cursor.execute('SELECT * FROM tag_score;')
    view = pd.DataFrame(cursor.fetchall())
    view.columns = cursor.column_names

    # write a csv file
    view.to_csv('DMA_project2_team%02d_part2_tag.csv' % team, sep=',', encoding='utf-8', index=False)
    # ------

    # TODO: Requirement 2-2. CREATE 2 VIEWS AND SAVE partial one to .csv file
    # User item rating view
    cursor.execute('''
    CREATE VIEW user_item_rating 
    AS SELECT user, item, SUM(cnt) AS rating
    FROM (
        (
        SELECT tag_name AS item, mentor_id AS user, 5*COUNT(*) AS cnt
        FROM tag_mentor AS tm, tag_score AS ts
        WHERE tm.tag_id = ts.tag_id
        GROUP BY tag_name, mentor_id
        )
        UNION ALL
        (
        SELECT tag_name AS item, mentee_id AS user, 5*COUNT(*) AS cnt
        FROM tag_mentee AS tm, tag_score AS ts
        WHERE tm.tag_id = ts.tag_id
        GROUP BY tag_name, mentee_id
        ) 
        UNION ALL
        (
        SELECT tag_name AS item, mentor_id AS user, LEAST(5,COUNT(*)) AS cnt
        FROM answer AS a, tag_score AS ts, tag_question AS tq
        WHERE ts.tag_id = tq.tag_id AND tq.question_id = a.question_id
        GROUP BY tag_name, mentor_id
        ) 
        UNION ALL
        (
        SELECT tag_name AS item, mentee_id AS user, LEAST(5,COUNT(*)) AS cnt
        FROM question AS q, tag_score AS ts, tag_question AS tq
        WHERE ts.tag_id = tq.tag_id AND tq.question_id = q.id
        GROUP BY tag_name, mentee_id
        )
        ) AS uir
    GROUP BY user, item;
    ''')

    # Partial user item rating view
    cursor.execute('''
    CREATE VIEW partial_user_item_rating
    AS SELECT user, item, rating
    FROM user_item_rating
    WHERE user IN (SELECT user FROM user_item_rating GROUP BY user HAVING COUNT(rating) > 3)
    ORDER BY user;
    ''')

    cursor.execute('SELECT * FROM partial_user_item_rating;')
    view = pd.DataFrame(cursor.fetchall())
    view.columns = cursor.column_names

    # write a csv file
    view.to_csv('DMA_project2_team%02d_part2_UIR.csv' % team, sep=',', encoding='utf-8', index=False)
    # ------
    cursor.close()

    # TODO: Requirement 2-3. MAKE HORIZONTAL VIEW
    df = view.copy()
    df = pd.concat([df['user'], pd.get_dummies(df['item'])], axis=1)
    hor_df = df.groupby(['user']).sum()

    # file name: DMA_project2_team##_part2_horizontal.pkl
    hor_df.to_pickle('DMA_project2_team%02d_part2_horizontal.pkl' % team)
    # ------
    
    # TODO: Requirement 2-4. ASSOCIATION ANALYSIS
    # filename: DMA_project2_team##_part2_association.pkl (pandas dataframe)
    frequent_itemsets = apriori(hor_df, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

    # write a pickle file
    rules.to_pickle('DMA_project2_team%02d_part2_association.pkl' % team)

    #### Use to Assessment ####
    # get a string from frozenset type
    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)).astype("unicode")
    # write a csv file
    rules.to_csv('DMA_project2_team%02d_part2_association.csv' % team, sep=',', index=False)


# TODO: Requirement 3-1. WRITE get_top_n
def get_top_n(algo, testset, id_list, n=10, user_based=True):
    results = defaultdict(list)
    if user_based:
        # TODO: testset의 데이터 중에 user id가 id_list 안에 있는 데이터만 따로 testset_id로 저장
        # Hint: testset은 (user_id, item_id, default_rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for i in range(len(testset)):
            if testset[i][0] in id_list:
                testset_id.append(testset[i])

        predictions = algo.test(testset_id)
        for uid, iid, true_r, est, _ in predictions:
            results[uid].append((iid, est))
            # TODO: results는 user_id를 key로,  [(item_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary
    else:
        # TODO: testset의 데이터 중 item id가 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장
        # Hint: testset은 (user_id, item_id, default_rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for i in range(len(testset)):
            if testset[i][1] in id_list:
                testset_id.append(testset[i])

        predictions = algo.test(testset_id)
        for uid, iid, true_r, est, _ in predictions:
            results[iid].append((uid, est))
            # TODO - results는 item_id를 key로, [(user_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary(3점)

    for id_, ratings in results.items():
        # TODO: rating 순서대로 정렬하고 top-n개만 유지
        results[id_] = sorted(ratings, key=lambda x: x[1], reverse=True)[:n]

    return results


# PART 3. Requirement 3-2, 3-3, 3-4
def part3():
    file_path = 'DMA_project2_team%02d_part2_UIR.csv' % team
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 10), skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)

    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()

    # TODO: Requirement 3-2. User-based Recommendation
    uid_list = ['ffffbe8d854a4a5a8ab1a381224f5b80',
                'ffe2f26d5c174e13b565d026e1d8c503',
                'ffdccaff893246519b64d76c3561d8c7',
                'ffdb001850984ce69c5f91360ac16e9c',
                'ffca7b070c9d41e98eba01d23a920d52']
    # TODO - set algorithm for 3-2-1
    algo = surprise.KNNBasic(k=40, min_k=1, sim_options={'name': 'cosine', 'user_based': True}, verbose=True)
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
    with open('3-2-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-10 results\n' % uid)
            for iid, score in ratings:
                f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 3-2-2
    algo = surprise.KNNWithMeans(k=40, min_k=1, sim_options={'name': 'pearson', 'user_based': True}, verbose=True)
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
    with open('3-2-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-10 results\n' % uid)
            for iid, score in ratings:
                f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - 3-2-3. Best Model
    kfold = KFold(n_splits=5, random_state=0)
    parameters = {'k': [30, 40, 50], 'min_k': [1],
                  'sim_options': {'name': ['pearson', 'cosine'], 'user_based': [True]}}

    # Select the best algo with grid search.
    print('Grid Search for user based model...')
    grid_KNNBasic = GridSearchCV(surprise.KNNBasic, measures=['rmse'], param_grid=parameters, cv=kfold)
    grid_KNNWithMeans = GridSearchCV(surprise.KNNWithMeans, measures=['rmse'], param_grid=parameters, cv=kfold)

    grid_KNNBasic.fit(data)
    grid_KNNWithMeans.fit(data)

    best_KNNBasic_score = grid_KNNBasic.best_score['rmse']
    best_KNNWithMeans_score = grid_KNNWithMeans.best_score['rmse']

    if best_KNNBasic_score < best_KNNWithMeans_score:
        algo_name = 'KNNBasic'
        best_algo_ub = grid_KNNBasic.best_estimator['rmse']
        with_parameters = grid_KNNBasic.best_params['rmse']
        score = best_KNNBasic_score

    else:
        algo_name = 'KNNWithMeans'
        best_algo_ub = grid_KNNWithMeans.best_estimator['rmse']
        with_parameters = grid_KNNWithMeans.best_params['rmse']
        score = best_KNNWithMeans_score

    print('The best UB algorithm is', algo_name, 'with', with_parameters, '\nscore:', score)

    # TODO: Requirement 3-3. Item-based Recommendation
    iid_list = ['art', 'teaching', 'career', 'college', 'medicine']
    # TODO - set algorithm for 3-3-1
    algo = surprise.KNNBasic(k=40, min_k=1, sim_options={'name': 'cosine', 'user_based': False}, verbose=True)
    algo.fit(trainset)
    results = get_top_n(algo, testset, iid_list, n=10, user_based=False)
    with open('3-3-1.txt', 'w') as f:
        for iid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Item ID %s top-10 results\n' % iid)
            for uid, score in ratings:
                f.write('User ID %s\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 3-3-2
    algo = surprise.KNNWithMeans(k=40, min_k=1, sim_options={'name': 'pearson', 'user_based': False}, verbose=True)
    algo.fit(trainset)
    results = get_top_n(algo, testset, iid_list, n=10, user_based=False)
    with open('3-3-2.txt', 'w') as f:
        for iid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Item ID %s top-10 results\n' % iid)
            for uid, score in ratings:
                f.write('User ID %s\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # TODO - 3-3-3. Best Model
    kfold = KFold(n_splits=5, random_state=0)
    parameters = {'k': [30, 40, 50], 'min_k': [1],
                  'sim_options': {'name': ['pearson', 'cosine'], 'user_based': [False]}}

    # Select the best algo with grid search.
    print('Grid Search for item based model...')
    grid_KNNBasic = GridSearchCV(surprise.KNNBasic, measures=['rmse'], param_grid=parameters, cv=kfold)
    grid_KNNWithMeans = GridSearchCV(surprise.KNNWithMeans, measures=['rmse'], param_grid=parameters, cv=kfold)

    grid_KNNBasic.fit(data)
    grid_KNNWithMeans.fit(data)

    best_KNNBasic_score = grid_KNNBasic.best_score['rmse']
    best_KNNWithMeans_score = grid_KNNWithMeans.best_score['rmse']

    if best_KNNBasic_score < best_KNNWithMeans_score:
        algo_name = 'KNNBasic'
        best_algo_ub = grid_KNNBasic.best_estimator['rmse']
        with_parameters = grid_KNNBasic.best_params['rmse']
        score = best_KNNBasic_score
    else:
        algo_name = 'KNNWithMeans'
        best_algo_ub = grid_KNNWithMeans.best_estimator['rmse']
        with_parameters = grid_KNNWithMeans.best_params['rmse']
        score = best_KNNWithMeans_score

    print('The best IB algorithm is', algo_name, 'with', with_parameters, '\nscore:', score)

    # TODO: Requirement 3-4. Matrix-factorization Recommendation
    # TODO - set algorithm for 3-4-1
    algo = surprise.SVD(n_factors=100, n_epochs=50, biased=False)
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
    with open('3-4-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-10 results\n' % uid)
            for iid, score in ratings:
                f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 3-4-2
    algo = surprise.SVD(n_factors=200, n_epochs=100, biased=True)
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
    with open('3-4-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-10 results\n' % uid)
            for iid, score in ratings:
                f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 3-4-3
    algo = surprise.SVDpp(n_factors=100, n_epochs=50)
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
    with open('3-4-3.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-10 results\n' % uid)
            for iid, score in ratings:
                f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 3-4-4
    algo = surprise.SVDpp(n_factors=100, n_epochs=100)
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
    with open('3-4-4.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-10 results\n' % uid)
            for iid, score in ratings:
                f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - 3-4-5. Best Model
    kfold = KFold(n_splits=5, random_state=0)
    parameters_SVD = {'n_factors': [50, 100, 200], 'n_epochs': [10, 50, 100, 200], 'biased': [True, False]}
    grid_SVD = GridSearchCV(surprise.SVD, measures=['rmse'], param_grid=parameters_SVD, cv=kfold)
    parameters_SVDpp = {'n_factors': [50, 100, 200], 'n_epochs': [10, 50, 100, 200]}
    grid_SVDpp = GridSearchCV(surprise.SVDpp, measures=['rmse'], param_grid=parameters_SVDpp, cv=kfold)

    grid_SVD.fit(data)
    grid_SVDpp.fit(data)

    best_SVD_score = grid_SVD.best_score['rmse']
    best_SVDpp_score = grid_SVDpp.best_score['rmse']

    if best_SVD_score < best_SVDpp_score:
        algo_name = 'SVD'
        best_algo_mf = grid_SVD.best_estimator['rmse']
        with_parameters = grid_SVD.best_params['rmse']
        score = best_SVD_score

    else:
        algo_name = 'SVDpp'
        best_algo_mf = grid_SVDpp.best_estimator['rmse']
        with_parameters = grid_SVDpp.best_params['rmse']
        score = best_SVDpp_score

    print('The best MF algorithm is', algo_name, 'with', with_parameters, '\nscore:', score)
    
    

if __name__ == '__main__':
    part1()
    part2()
    part3()




