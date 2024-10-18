# AI-Assistant
Zach Gunderson and Aren Desai attempt to make AGI.

## Overview
See [the project description docs.](https://docs.google.com/document/d/1LAKMdX9D1TIlcan3xt2AHcYOopl7pGLNz-sFh7Ijrn0/edit)

## Design Paradigm
OOP-approach. Thinking we have one main loop, running permanently and processing functions. A user can use the default functions and add plugins, which are python functions, by specifying them in a configuration file. Designing new functions for the AIA to call should be easy. 

## LLM Communication
We'll use gRPC. Communication is built in ```llm.proto```, which is compiled with this command:
```
python3 -m grpc_tools.protoc -I=. --python_out=. --grpc_p^Chon_out=. llm.proto
```

The outputs, ```llm_pb2_grpc.py``` and ```llm_pb2.py```, are placed in the ```server``` directory, but also are needed wherever a client needs to communicate with the server. 
In the ```server``` directory contains a Dockerfile, which can be spun up and queried on port ```5440``` with gRPC for the LLM responses. 
