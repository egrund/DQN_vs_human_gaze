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

def start_server_sender(port, buffer, batch_size, indices,can_sample):

    soc = create_socket(port)
    while True:
        con,_ = soc.accept()
        with con:
            while not can_sample["0"]:
                pass
            while indices["0"] == None:
                pass
            data = pickle.dumps(buffer.sample_minibatch(indices["0"][batch_size*(port-8000):batch_size*(port-8000+1)]))
            con.sendall(data)
    
def start_server_indice_selector(port,buffer, indices, batch_size, inner_its, can_sample):
    soc = create_socket(port)
    while True:
        con,_ = soc.accept()
        while not can_sample["0"]:
                pass
        indices["0"] = buffer.get_indices(batch_size*inner_its)
        while True:
            message = con.recv(4096)
            if not message: break
        indices["0"] = None

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

        indices = manager.dict()
        indices["0"] = None

        # crate servers for loading the minibatches
        running_processes = []
        for i in range(params["INNER_ITS"]):
            p = Process(target = start_server_sender, args = (8000+i,buffer, params["BATCH_SIZE"], indices ,can_sample))
            p.start()
            running_processes.append(p)

        # create server for receiving new data points and priority updates
        p = Process(target = start_server_data_receiver, args = (7999, buffer,can_sample))
        p.start()
        running_processes.append(p)

        # sampling minibatches has to be done in the scope of the connection to this server
        p = Process(target = start_server_indice_selector, args = (7997, buffer,indices, params["BATCH_SIZE"], params["INNER_ITS"], can_sample))
        p.start()
        running_processes.append(p)

        print("server started")
        
        a = input("press any key to restart")
        for p in running_processes:
            p.kill()
   
    
