'''
Driver code for SMART_CyberBook
It's main() is invoked when "Run SMART" is triggered by the service.
The main() invokes the following features in sequential order:
    1. Fetch instructionalText ids and instructionalText texts
    2. Run SMART_CORE on the fetched data to generate skills
    3. Update database table with the skill names.
'''
from mysql.connector import connect, Error
from sshtunnel import SSHTunnelForwarder
import mysql.connector
import sshtunnel

from .db import DB
from .ssh import SSH
from .text_cleaner import clean_text
from SMART_CORE.text1_skill_mapping import text1_skill_mapping

def main(course_id, smart_hyperparameters):    
    print('Running SMART...')
    db_obj = DB()
    db_obj.ssh_set_credentials(
        ssh = SSH(host='10.154.30.212', user='iec', password='iec iec'),
        host='127.0.0.1',
        port='3506',
        db="edxapp_csmh",
        user='root',
        password=''
        )
    
    table = "export_course_content_and_skill_validation"
    course_id_condition = "course_id = '{0}'".format(course_id)
    print(course_id_condition)

    ## fetch problem id
    data = db_obj.get_data(table, ['problem_name'], course_id_condition)
    if data["status"] == "success":
        result = data["result"]
        text_ids = [row['problem_name'] for row in result]
    else:
        print(data["status"])
        
    ## fetch problem text, answer, hint
    data = db_obj.get_data(table, ['question', 'correct_answer', 'hint'], course_id_condition)
    if data["status"] == "success":
        result = data["result"]
        texts = clean_text(result)
    else:
        print(data["status"])

    ## run smart
    problem_name_to_skillname = text1_skill_mapping(text_ids, texts, smart_hyperparameters, is_called_from_service = True)
    print('Generated skills: Success')

    ## update table
    column_names = ('problem_name', 'skillname')
    result = db_obj.update_values_from_mappings(table, column_names, problem_name_to_skillname, course_id_condition)
    print(result)
    

if __name__ == '__main__':
    main()
    