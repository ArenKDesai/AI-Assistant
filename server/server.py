import llm_pb2, llm_pb2_grpc, grpc
from datetime import datetime
from concurrent import futures

class LLM(llm_pb2_grpc.LLMServicer):
    """
    Main class for communication with the LLM.

    process_query is necessary for basic call/response, 
    but we could add more communication protocols for 
    stuff like uploading files. 
    """
    def process_query(self, request, context):
        query = request.request

        # TODO: This is where we will query the LLM
        # Also need to decide how to process 
        response = "Not Implemented"

        return llm_pb2.QueryResp(response=response)

if __name__ == '__main__':
    print(f"Starting server on {datetime.now().ctime()}")
    # NOTE: If we want more threads, we can increment max_workers
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[("grpc.so_reuseport", 0)])
    # TODO: Figure out how to handle port communication
    port_num = "0.0.0.0:5440"
    server.add_insecure_port(port_num)

    server.start()
    server.wait_for_termination()
