
from milvus import default_server
from milvus import MilvusServer
from milvus import MilvusServerConfig


from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

config = MilvusServerConfig(proxy_port=19531, data_dir=r'd:\milvus_data')
server = MilvusServer(config=config)
server.start()
server.wait()

utility.list_users()


