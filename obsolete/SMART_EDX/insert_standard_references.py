"""
Insert TEKS reference into the
MySQL Database
"""

import pymysql as db


def create_connection(paragraphs_to_teks, questions_to_teks):
    user = 'root'
    password = ''
    database = 'edxapp_csmh'

    conn = db.connect(host="127.0.0.1",
                      port=3306,
                      user=user,
                      passwd=password,
                      db=database)

    run_query(conn, paragraphs_to_teks, questions_to_teks)

    conn.close()


def run_query(conn, paragraphs_to_teks, questions_to_teks):

    cur = conn.cursor()

    for xblock in paragraphs_to_teks:
        teks = xblock[2]
        cur.execute('UPDATE edxapp_csmh.export_course_content_and_skill_validation '
                    'SET TEKS_reference = %s '
                    'WHERE id = %s;', (teks, xblock[0]))

        conn.commit()

    for xblock in questions_to_teks:
        teks = xblock[2]
        cur.execute('UPDATE edxapp_csmh.export_course_content_and_skill_validation '
                    'SET TEKS_reference = %s '
                    'WHERE id = %s;', (teks, xblock[0]))

        conn.commit()


def insert_TEKS(paragraphs_to_teks, questions_to_teks):
    create_connection(paragraphs_to_teks, questions_to_teks)
