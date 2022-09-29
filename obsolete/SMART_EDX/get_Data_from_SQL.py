"""
Retrieve Contents of the course
from MySQL database
"""

import pymysql as db


def create_connection():
    user = 'root'
    password = ''
    database = 'edxapp_csmh'

    conn = db.connect(host="127.0.0.1",
                      port=3306,
                      user=user,
                      passwd=password,
                      db=database)

    result = run_query(conn)
    conn.close()
    return result


def run_query(conn):

    cur = conn.cursor()
    cur.execute("SELECT * FROM "
                "export_course_content_and_skill_validation;")

    result = cur.fetchall()

    return result


def get_content():

    result = create_connection()

    questions = []
    paragraphs = []

    for row in result:
        if row[1] == 'course-v1:University+CS101+2015_T1':

            if row[6] == 'TextParagraph':
                text = row[9]
                paragraphs.append((row[0], text))
            else:
                text = row[10]
                questions.append((row[0], text))

    return paragraphs, questions
