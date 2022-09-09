import socket
from multiprocessing import Process
import os
import pickle
from buffer import Buffer, BufferManager
from multiprocessing.managers import SyncManager

def create_socket(port):

    soc = socket.socket()
    soc.bind(('localhost',port))
    soc.listen(1)

    return soc

def start_server_sender(port, buffer, batch_size, can_sample):

    soc = create_socket(port)
    while True:
        con,_ = soc.accept()
        with con:
            while not can_sample["0"]:
                pass
            data = pickle.dumps(buffer.sample_minibatch(batch_size))
            con.sendall(data)
    

def start_server_data_receiver(port, buffer, can_sample):

    soc = create_socket(port)
    while True:
        fragments = []
        con,_ = soc.accept()
        can_sample["0"] = False
        while True:
            recvfile = con.recv(4096)
            if not recvfile: break
            fragments.append(recvfile)

        data = pickle.loads(b''.join(fragments))

        if data["task"] == "extend":
            buffer.extend(data["values"])
        elif data["task"] == "priority":
            buffer.update_priority(data["values"])
        can_sample["0"] = True

if __name__ == "__main__":

    while True:
        
        print("Waiting for hyperparameters")
        # start server for receiving hyperparameters
        soc = create_socket(7998)
        con,_ = soc.accept()
        fragments = []
        while True:
            recvfile = con.recv(4096)
            if not recvfile: break
            fragments.append(recvfile)
        params = pickle.loads(b''.join(fragments))

        print("Received hyperparameters")

        BufferManager.register("Buffer", Buffer)
        manager = BufferManager()
        manager.start()

        # create buffer
        buffer = manager.Buffer(params["BUFFER_SIZE"], params["BUFFER_MIN"], params["BUFFER_PATH"], params["KEEP_IN_MEM"], params["USE_PRIORITIZED_REPLAY"])
        buffer.clear()

        can_sample = manager.dict()
        can_sample["0"] = False
        # crate servers for loading the minibatches
        running_processes = []
        for i in range(params["INNER_ITS"]):
            p = Process(target = start_server_sender, args = (8000+i,buffer, params["BATCH_SIZE"],can_sample))
            p.start()
            running_processes.append(p)

        # create server for receiving new data points and priority updates
        p = Process(target = start_server_data_receiver, args = (7999, buffer,can_sample))
        p.start()
        running_processes.append(p)

        print("server started")
        
        a = input("press any key to restart")
        for p in running_processes:
            p.kill()
   
    
