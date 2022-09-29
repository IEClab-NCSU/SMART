"""
Insert Skills into the
MySQL Database
"""

import pymysql as db


def create_connection(paragraphs_to_skills, questions_to_skills):
    user = 'root'
    password = ''
    database = 'edxapp_csmh'

    conn = db.connect(host="127.0.0.1",
                      port=3306,
                      user=user,
                      passwd=password,
                      db=database)
    run_query(conn, paragraphs_to_skills, questions_to_skills)

    conn.close()


def run_query(conn, paragraphs_to_skills, questions_to_skills):

    cur = conn.cursor()

    for xblock in paragraphs_to_skills:

        if type(xblock[1]) == list:
            skills = ', '.join(xblock[1])
            cur.execute('UPDATE edxapp_csmh.export_course_content_and_skill_validation '
                        'SET SMART_SkillName = %s '
                        'WHERE id = %s;', (skills, xblock[0]))

            conn.commit()

    for xblock in questions_to_skills:

        if type(xblock[1]) == list:
            skills = ', '.join(xblock[1])
            cur.execute('UPDATE edxapp_csmh.export_course_content_and_skill_validation '
                        'SET SMART_SkillName = %s '
                        'WHERE id = %s;', (skills, xblock[0]))

            conn.commit()


def insert_skills(paragraph_to_skills, question_to_skills):
    create_connection(paragraph_to_skills, question_to_skills)
