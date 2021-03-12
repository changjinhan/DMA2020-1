import mysql.connector

# TODO: REPLACE THE VALUE OF VARIABLE team (EX. TEAM 1 --> team = 1)
team = 3

# Requirement1: create schema ( name: DMA_team## )
def requirement1(host, user, password):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    
    # TODO: WRITE CODE HERE
    print('Creating schema...')
    cursor.execute('DROP DATABASE IF EXISTS DMA_team10;')
    cursor.execute('CREATE DATABASE IF NOT EXISTS DMA_team10;')

    # TODO: WRITE CODE HERE
    cnx.close()
    
# Requierement2: create table
def requirement2(host, user, password):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    
    # TODO: WRITE CODE HERE
    print('Creating tables...')
    cursor.execute('USE DMA_team03;')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mentor(
    id VARCHAR(255) NOT NULL, 
    district VARCHAR(255),
    field VARCHAR(255),
    introduction VARCHAR(255),
    joined_date DATETIME NOT NULL,
    PRIMARY KEY (id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mentee(
    id VARCHAR(255) NOT NULL,
    district VARCHAR(255),
    joined_Date DATETIME NOT NULL,
    PRIMARY KEY (id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS question(
    id VARCHAR(255) NOT NULL,
    mentee_id VARCHAR(255) NOT NULL,
    posted_date DATETIME NOT NULL,
    title VARCHAR(255) NOT NULL,
    body INT(11) NOT NULL,
    score INT(11) NOT NULL,
    PRIMARY KEY (id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS answer(
    id VARCHAR(255) NOT NULL,
    mentor_id VARCHAR(255) NOT NULL,
    question_id VARCHAR(255) NOT NULL,
    answered_date DATETIME NOT NULL,
    body INT(11) NOT NULL,
    score INT(11) NOT NULL,
    PRIMARY KEY (id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS comment(
    question_id VARCHAR(255) NOT NULL,
    comment_order INT(11) NOT NULL,
    comment_date DATETIME NOT NULL,
    body INT(11) NOT NULL,
    PRIMARY KEY (question_id, comment_order) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tag(
    id INT(11) NOT NULL,
    name VARCHAR(255),
    PRIMARY KEY (id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tag_mentee(
    tag_id INT(11) NOT NULL,
    mentee_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (tag_id, mentee_id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tag_mentor(
    tag_id INT(11) NOT NULL,
    mentor_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (tag_id, mentor_id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tag_question(
    tag_id INT(11) NOT NULL,
    question_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (tag_id, question_id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mentoring_group(
    id VARCHAR(255) NOT NULL,
    group_type INT(11) NOT NULL,
    need_allow TINYINT(1) NOT NULL,
    openness TINYINT(1) NOT NULL,
    mentor VARCHAR(255) NOT NULL,
    created_date DATETIME NOT NULL,
    PRIMARY KEY (id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS group_membership(
    group_id VARCHAR(255) NOT NULL,
    mentee_id VARCHAR(255) NOT NULL,
    group_joined_date DATETIME NOT NULL,
    PRIMARY KEY (group_id, mentee_id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS email(
    id INT(11) NOT NULL,
    recipient_id VARCHAR(255) NOT NULL,
    date_sent DATETIME NOT NULL,
    frequency_level VARCHAR(255) NOT NULL,
    PRIMARY KEY (id) );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS email_question(
    email_id INT(11) NOT NULL,
    question_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (email_id, question_id) );
    ''')

    # TODO: WRITE CODE HERE
    cnx.close()
    
# Requirement3: insert data
def requirement3(host, user, password, directory):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')

    # TODO: WRITE CODE HERE
    print('Inserting data...')
    cursor.execute('USE DMA_team03;')
    table_name = ['mentor', 'mentee', 'question', 'answer', 'comment', 'tag', 'tag_mentee',
                  'tag_mentor', 'tag_question', 'mentoring_group', 'group_membership',
                  'email', 'email_question']

    for table in table_name:
        filepath = directory + '/' + table + '.csv'
        with open(filepath, 'r') as csv_data:
            next(csv_data, None) #skip the headers
            if table == 'mentor':
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table + ' VALUES (%s,%s,%s,%s,%s)', row)
            elif table == 'mentee':
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table +' VALUES (%s,%s,%s)', row)
            elif table in ['question', 'answer']:
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table +' VALUES (%s,%s,%s,%s,%s,%s)', row)
            elif table == 'comment':
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table +' VALUES (%s,%s,%s,%s)', row)
            elif table in ['tag', 'tag_mentee', 'tag_mentor', 'tag_question']:
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table +' VALUES (%s,%s)', row)
            elif table == 'mentoring_group':
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table +' VALUES (%s,%s,%s,%s,%s,%s)', row)
            elif table == 'group_membership':
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table +' VALUES (%s,%s,%s)', row)
            elif table == 'email':
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table +' VALUES (%s,%s,%s,%s)', row)
            elif table == 'email_question':
                for row in csv_data:
                    # Change the null data
                    row = row.strip().split(sep=',')
                    if '' in row:
                        temp = []
                        for item in row:
                            if item == '':
                                item = None
                            temp.append(item)
                        row = temp
                    cursor.execute('INSERT INTO ' + table +' VALUES (%s,%s)', row)

    # TODO: WRITE CODE HERE
    cnx.commit()
    cnx.close()
    
# Requirement4: add constraint (foreign key)
def requirement4(host, user, password):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')

    
    # TODO: WRITE CODE HERE
    print('Adding constraints...')
    cursor.execute('USE DMA_team03;')
    cursor.execute('ALTER TABLE question ADD CONSTRAINT FOREIGN KEY (mentee_id) REFERENCES mentee(id);')
    cursor.execute('ALTER TABLE answer ADD CONSTRAINT FOREIGN KEY (mentor_id) REFERENCES mentor(id);')
    cursor.execute('ALTER TABLE answer ADD CONSTRAINT FOREIGN KEY (question_id) REFERENCES question(id);')
    cursor.execute('ALTER TABLE comment ADD CONSTRAINT FOREIGN KEY (question_id) REFERENCES question(id);')
    cursor.execute('ALTER TABLE tag_mentee ADD CONSTRAINT FOREIGN KEY (tag_id) REFERENCES tag(id);')
    cursor.execute('ALTER TABLE tag_mentee ADD CONSTRAINT FOREIGN KEY (mentee_id) REFERENCES mentee(id);')
    cursor.execute('ALTER TABLE tag_mentor ADD CONSTRAINT FOREIGN KEY (tag_id) REFERENCES tag(id);')
    cursor.execute('ALTER TABLE tag_mentor ADD CONSTRAINT FOREIGN KEY (mentor_id) REFERENCES mentor(id);')
    cursor.execute('ALTER TABLE tag_question ADD CONSTRAINT FOREIGN KEY (tag_id) REFERENCES tag(id);')
    cursor.execute('ALTER TABLE tag_question ADD CONSTRAINT FOREIGN KEY (question_id) REFERENCES question(id);')
    cursor.execute('ALTER TABLE mentoring_group ADD CONSTRAINT FOREIGN KEY (mentor) REFERENCES mentor(id);')
    cursor.execute('ALTER TABLE group_membership ADD CONSTRAINT FOREIGN KEY (group_id) REFERENCES mentoring_group(id);')
    cursor.execute('ALTER TABLE group_membership ADD CONSTRAINT FOREIGN KEY (mentee_id) REFERENCES mentee(id);')
    cursor.execute('ALTER TABLE email ADD CONSTRAINT FOREIGN KEY (recipient_id) REFERENCES mentor(id);')
    cursor.execute('ALTER TABLE email_question ADD CONSTRAINT FOREIGN KEY (email_id) REFERENCES email(id);')
    cursor.execute('ALTER TABLE email_question ADD CONSTRAINT FOREIGN KEY (question_id) REFERENCES question(id);')

    # TODO: WRITE CODE HERE
    cnx.close()
    
# TODO: REPLACE THE VALUES OF FOLLOWING VARIABLES
host = 'localhost'
user = 'root'
password = '1234'
directory_in = '/Users/changjinhan/DMA_project1/dataset'


requirement1(host=host, user=user, password=password)
requirement2(host=host, user=user, password=password)
requirement3(host=host, user=user, password=password, directory=directory_in)
requirement4(host=host, user=user, password=password)
print('Done!')






