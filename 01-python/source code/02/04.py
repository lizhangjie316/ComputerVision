from queue import Queue
import threading


def job(l,q):
    for i in range(len(l)):
        l[i] = l[i]**2
    q.put(l)
    

def multithreading():
    q = Queue()    # 此处的Queue起一个携带出所有的结果的作用
    threads = []
    data = [[1,2,3],[3,4,5],[1,3,5],[2,4,6]]
    for i in range(len(data)):
        t = threading.Thread(target=job,args=(data[i],q))
        t.start()
        threads.append(t)

    for thread in threads:  #待所有线程全都执行完成后
        thread.join()

    #此时所有数据均已经存储进入q中

    results = []

    for i in range(4):
        results.append(q.get())

    print(results)

multithreading()

    
