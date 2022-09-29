# import MySQLdb # python2
from mysql.connector import connect, Error #python3

from sshtunnel import SSHTunnelForwarder


class DB:

    def __init__(self):
        self.ssh = None
        self.host = None
        self.user = None
        self.password = None
        self.cursor = None
        self.server = None
        self.connection = None

    def ssh_set_credentials(self, ssh=None, host=None, port=None, db=None, user=None, password=None):
        self.ssh = ssh
        self.host = host
        self.port = port
        self.db = db
        self.user = user
        self.password = password

    def ssh_tunnel(self):
        self.server = SSHTunnelForwarder(
            (self.ssh.host, 22),
            ssh_username=self.ssh.user,
            ssh_password=self.ssh.password,
            remote_bind_address=(self.host, int(self.port))
        )

    def connect(self):
        port = int(self.port)
        host = self.host
        if self.ssh is not None:
            self.ssh_tunnel()
            self.server.start()
            port = self.server.local_bind_port
            host = '0.0.0.0'

        # self.connection = MySQLdb.connect(
        #                 user = self.user,
        #                 password = self.password,
        #                 host = host,
        #                 database = self.db,
        #                 port = port)
        try:
            self.connection = connect(
                user=self.user,
                password=self.password,
                host='localhost',
                database = self.db,
                port= port,
                use_pure = True
            )
        except Error as e:
            print('ERROR block')
            print(e)


    def close(self):
        self.connection.close()
        if self.ssh is not None:
            self.server.stop()

    def get_data(self, table, columns, condition):
        self.connect()
        cursor = self.connection.cursor()
        sql = """SELECT {0} FROM {1} WHERE {2}""".format(','.join(columns), table, condition)
        result = []
        try:
            cursor.execute(sql)
            if columns != ["*"]:
                for row in cursor:
                    data = {}
                    for i in range(len(columns)):
                        data[columns[i]] = row[i]
                    result.append(data)
            else:
                for row in cursor:
                    result.append(row)
            self.close()
        except Exception as e:
            self.close()
            return {"status":"error","exception":str(e)}
        return {"status":"success","result":result}


    def get_count(self, table, condition):
        self.connect()
        cursor = self.connection.cursor()
        sql = """SELECT COUNT(*) FROM {0} WHERE {1}""".format(table, condition)
        result = {}
        try:
            cursor.execute(sql)
            row = cursor.fetchone()
            result['count'] = row[0]
            self.close()
        except Exception as e:
            self.close()
            return {"status":"error","exception":str(e)}
        return {"status":"success","result":result}


    def insert_values(self, table, data):
        self.connect()
        columns = data.keys()
        values = []
        for c in columns:
            values.append("'"+str(data[c])+"'")
        cursor = self.connection.cursor()
        sql = """INSERT INTO {0} ({1}) VALUES ({2})""".format(table, ','.join(columns), ','.join(values))
        result = {}
        try:
            cursor.execute(sql)
            self.connection.commit()
            self.close()
            return {"status":"success","result":"Inserted"}
        except Exception as e:
            self.connection.rollback()
            self.close()
            return {"status":"error","exception":str(e)}


    def update_values(self, table, data, condition):
        self.connect()
        columns = data.keys()
        set_list = []
        set_str = ""
        for c in columns:
            set_list.append(c+"='"+str(data[c])+"'")
        set_str = ','.join(set_list)
        sql = """UPDATE {0} SET {1} WHERE {2}""".format(table, set_str, condition)
        result = {}
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            self.connection.commit()
            self.close()
            return {"status":"success","result":"Updated"}
        except Exception as e:
            self.connection.rollback()
            self.close()
            return {"status":"error","exception":str(e)}
    
    def update_values_from_mappings(self, table, column_names, column1_to_column2, condition):
        column1_name, column2_name = column_names

        self.connect()
        cursor = self.connection.cursor()
        for column1_value, column2_value in column1_to_column2.items():
            data = {column2_name:column2_value}
            mapping_condition = "{0}='{1}'".format(column1_name, column1_value)
            combined_condition = condition + "AND " + mapping_condition

            columns = data.keys() # columns to update
            set_list = []
            for c in columns:
                set_list.append(c+"='"+str(data[c])+"'")
                # set_list.append(c + "=" +"'dummy'")
            set_str = ','.join(set_list)

            sql = """UPDATE {0} SET {1} WHERE {2}""".format(table, set_str, combined_condition)
            try:
                cursor.execute(sql) 
                # print(sql)
            except Exception as e:
                self.connection.rollback()
                self.close()
                return {"status":"error","exception":str(e)}

        try:
            self.connection.commit()
            return {"status":"success","result":"Updated"}
        except Exception as e:
            self.connection.rollback()
            return {"status":"error","exception":str(e)}
        finally:
            self.close()

        
    def delete_rows(self, table, condition):
        self.connect()
        sql = """DELETE FROM {0} WHERE {1}""".format(table, condition)
        result = {}
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            self.connection.commit()
            self.close()
            return {"status":"success","result":"Deleted"}
        except Exception as e:
            self.connection.rollback()
            self.close()
            return {"status":"error","exception":str(e)}

