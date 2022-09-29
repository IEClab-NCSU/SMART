import os
from getpass import getpass
from mysql.connector import connect, Error
from sshtunnel import SSHTunnelForwarder

from db import DB
import mysql.connector
import sshtunnel

def connect_ssh():    
    _host, _ssh_port = (os.environ['OPENEDX_HOST'], os.environ['OPENEDX_SSH_PORT'])
    _remote_bind_address = '127.0.0.1'
    _remote_mysql_port = os.environ['OPENEDX_SSH_PORT']
    _local_bind_address = 'localhost'
    _local_mysql_port = os.environ['OPENEDX_SSH_PORT']
    connection=12
    with sshtunnel.SSHTunnelForwarder(
            (_host, int(_ssh_port)),
            ssh_username=os.environ['OPENEDX_SSH_USER'],
            ssh_password=os.environ['OPENEDX_SSH_PASS'],
            remote_bind_address=(_remote_bind_address, int(_remote_mysql_port)),
            local_bind_address=(_local_bind_address, int(_local_mysql_port))
    ) as tunnel:    
        connection = mysql.connector.connect(
            user=os.environ['OPENEDX_DB_USER'],
            password=os.environ['OPENEDX_DB_PASS'],
            host=_local_bind_address,
            database='edxapp_csmh',
            port=int(_local_mysql_port),
            use_pure = True)
    
        cursor = connection.cursor()
        query  = "Select * from export_course_content_and_skill_validation"
        cursor.execute(query)
        rows = cursor.fetchall()
        print(len(rows))
        # print(connection)

if __name__ == '__main__':
    # host = '10.154.30.212'
    # # port='3306'
    # db_name = 'edxapp_csmh'
    # user='root'
    # password = ''
    # db_obj = DB()
    # db_obj.ssh_set_credentials(host=host, port=port, db=db_name, user=user, password=password)
    # db_obj.connect()
    # create_connection(host, port, db_name, user, password)
    connect_ssh()
    