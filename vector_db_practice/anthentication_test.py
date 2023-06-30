# here we tried the username and password

from pymilvus import utility
from pymilvus import connections
from pymilvus.settings import Config


def set_new_user(username: str, password: str):
    utility.create_user(username, password, using='default')

# connection to the server
#connections.connect(host='127.0.0.1', port=19531)

# set the username: plan, password: plan001
#set_new_user('plan', 'plan001')

#connections.disconnect(alias=Config.MILVUS_CONN_ALIAS)


# use password to connect the server
#milvus-server --debug --data d:\milvus_data  --system-log-level debug \
# --proxy-port 19531 --authorization-enabled  false
connections.connect(host='127.0.0.1', port=19531, user='plan', password='plan001')

# delete the user
utility.delete_user('plan', using='default')

# close the disconnect
connections.disconnect(alias=Config.MILVUS_CONN_ALIAS)

