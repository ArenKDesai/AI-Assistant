import sys
from concurrent import futures
import grpc, llm_pb2, llm_pb2_grpc

port_num = "5440"

class QueryLLM(llm_pb2_grpc.QueryLLM):
    def QueryReq(self, request, context):
        print(f"Query Request: {request.request}")

if __name__ == '__main__':
    print("Server starting.")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[("grpc.so_reuseport", 0)])
    llm_pb2_grpc.add_QueryLLMServicer_to_server(QueryLLM(), server)
    server.add_insecure_port(f"0.0.0.0:{port_num}")
    server.start()
    server.wait_for_termination()
