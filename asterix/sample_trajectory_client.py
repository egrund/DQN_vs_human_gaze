import socket
import pickle
import queue

def create_trajectory_client_send(data):
    soc = socket.socket()
    connected = False
    while not connected:
        try: 
            soc.connect(("localhost",7994))
            connected = True
        except:
            pass
    with soc:
        soc.sendall(pickle.dumps(data))



def create_trajectory_client_receive(q):
    soc = socket.socket()
    connected = False
    fragments = []
    while not connected:
        try: 
            soc.connect(("localhost",7994))
            connected = True
        except:
            pass
    with soc:
        while True:
            fragment = soc.recv(4096)
            if not fragment : break
            fragments.append(fragment)
        data = pickle.loads(b''.join(fragments))
    
    q.put(data)


if __name__ == "__main__":
    soc = socket.socket()
    soc.bind(("localhost", 7995))
    soc.listen(1)

    while True:
        print("waiting...")
        con,_ = soc.accept()
        print("accepted receive")
        fragments = []
        with con:
            while True:
                fragment = con.recv(4096)
                if not fragment : break
                fragments.append(fragment)
        
        data = pickle.loads(b''.join(fragments))

        weights = data["weights"]
        batch = data["batch"]
        epsilon = data["epsilon"]
        env_name = data["env_name"]
        frame_skips = data["frame_skips"]
        imgx = data["imgx"]
        imgy = data["imgy"]
        print("sending...")
        create_trajectory_client_send(data)

        q = queue.Queue()
        print("receiving...")
        create_trajectory_client_receive(q)
        new_data = q.get(block = True)
        print("waiting...")
        con,_ = soc.accept()
        print("accepted send")
        with con:
            con.sendall(pickle.dumps(new_data))