
Here  we try milvus and log some practice.

# 1 start up the milvus server in development env.

> milvus-server --debug --data d:\milvus_data  --system-log-level debug \
> --proxy-port 19531 --authorization-enabled  false

--debug: parameter indicates we use debug mode.

--data: specify the local disk path, used to store the log, index file etc.

--system-log-level: set the log level

--proxy-port: specify the listen port for connection.


If have set the user and password, next we startup the server, we could add the below parameter.

--authorization-enabled  true


# 2  