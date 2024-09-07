import socket
import tqdm
import os
import time
import threading
import re

def received():


    # 接收客户端信息
    received = client_socket.recv(Buffersize).decode(errors='ignore')
    filename, file_size = received.split(SEPARATOR)
    # 获取文件的名字,大小
    filename = os.path.basename(filename)
    #print(re.findall(r'\d+',file_size))

    #file_size = int(file_size)

    file_size=re.findall(r'\d+',file_size)
    file_size =int(file_size[0])


    # 文件接收处理
    progress = tqdm.tqdm(range(file_size), f'接收{filename}', unit='B', unit_divisor=1024, unit_scale=True)

    with open('./rec/'+filename, 'wb') as f:

        for _ in progress:
            # 从客户端读取数据
            bytes_read = client_socket.recv(Buffersize)
            # 如果没有数据传输内容
            if not bytes_read:
                break
            # 读取写入
            f.write(bytes_read)
            # 更新进度条
            progress.update(len(bytes_read))
            #time.sleep(0.001)



if __name__ == '__main__':
    msg = 'done.'

    # 设置服务器的ip和 port
    # 服务器信息
    sever_host = '127.0.0.1'
    #sever_host = '192.168.1.102'

    sever_port = 2234
    # 传输数据间隔符
    SEPARATOR = '<SEPARATOR>'

    # 文件缓冲区
    Buffersize = 4096*10

    s = socket.socket()
    s.bind((sever_host, sever_port))

    # 设置监听数
    s.listen(1)
    print(f'服务器监听{sever_host}:{sever_port}')

    while 1:
        # 接收客户端连接
        client_socket, address = s.accept()
        # 打印客户端ip
        print(f'客户端{address}连接')

        received()

        client_socket.send(msg.encode())
        #print(msg)

    # 关闭资源
    client_socket.close()
    s.close()


