from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

conn = connections.connect(host='127.0.0.1', port=19531, user='root', password='Milvus')

print(utility.list_users(include_role_info=True))

users = utility.list_usernames(using='default')


print(users)