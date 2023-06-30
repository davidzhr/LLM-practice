# This  example is from https://milvus.io/docs/example_code.md .
# when i try this example, i have not set the password for milvus server.
# so if you have set a user and password, you should use below format to connect the server.
# connections.connect(
#   alias="default",
#   user='username',
#   password='password',
#   host='localhost',
#   port='19530'
# )


from pymilvus import (
    default_server,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import random

default_server.set_base_dir('milvus_data')

# connect the local milvus server, here the port is 15931, you should know the default port is 15930. 
connections.connect(host='127.0.0.1', port=19531)
print(utility.get_server_version())

# define the Collection Schema
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
]

# create the Collection
schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
hello_milvus = Collection("hello_milvus", schema)

# construct the testing data
entities = [
    [i for i in range(30)],  # field pk
    [float(random.randrange(-20, -10)) for _ in range(30)],  # field random
    ["q{}".format(i) for i in range(30)],
    ["a{}".format(i) for i in range(30)],
    [[random.random() for _ in range(8)] for _ in range(30)],  # field embeddings
]

print(len(entities), entities)
insert_result = hello_milvus.insert(entities)
print(insert_result)
hello_milvus.flush()

index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
hello_milvus.create_index("embeddings", index)


hello_milvus.load()

vectors_to_search = entities[-1][-2:]
print(f'vectors_to_search: {vectors_to_search}')

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3,
                             output_fields=["random", 'question', 'answer'])
print(result)


# delete the first entity
expr = f"pk in [{entities[0][0]}]"
hello_milvus.delete(expr)

# drop the collection
utility.drop_collection("hello_milvus")

