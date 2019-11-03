import threading
import time

def thread_job():
   # print('This is a added Thread, number is %s' % threading.current_thread())
    print('T1 start\n')
    for i in range(10):
       time.sleep(0.1)

    print('T1 finish')

#thread_job()

def main():
    added_thread = threading.Thread(target=thread_job,name='T1')  #线程的任务是thread_job
    added_thread.start()
   # print('Thread count:',threading.active_count())
   # print('list the thread:',threading.enumerate())
   # print('list the thread now:',threading.current_thread())

    added_thread.join()
    print('all done')

if __name__ == '__main__':   #不太懂这块是什么   __main__   为当前执行文件的名称(带.py)
    main()                  #    __name__  在当前模块中为文件名(带有.py);当被import到其它文件中时，__name__ 等于被引模块名称(不带.py)            
