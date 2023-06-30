from pymilvus import connections
from pymilvus import db

# create the connection
conn = connections.connect(host="127.0.0.1", port=19531)

# create the db
database = db.create_database("books")

# use the db
db.using_database("books")

# list the dbs
db.list_database()

# drop the db
db.drop_database("books")
db.list_database()



_HOST = '127.0.0.1'
_PORT = '19531'
_ROOT = "root"
_ROOT_PASSWORD = "Milvus"
_ROLE_NAME = "test_role"
_PRIVILEGE_INSERT = "Insert"


def connect_to_milvus(db_name="default"):
    print(f"connect to milvus\n")
    connections.connect(host=_HOST, port=_PORT, db_name=db_name)


