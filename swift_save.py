import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import os
from turbojpeg import TurboJPEG

def save_img_proc(control_q):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_dll = os.path.join( dir_path , R"turbojpeg.dll")
    jpeg_engine = TurboJPEG(path_dll)
    while True:
        try:
            item = control_q.get()
            shm = shared_memory.SharedMemory(item[0])
            img_shape = item[1]
            img = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
            fn = item[2]
            with open(fn, 'wb') as out_file:
                out_file.write(jpeg_engine.encode(img, quality=97))
        except:
            print('Error in save_img_proc()')


class Swift_save():

    def __init__(self):
        n_save_proc = 3
        self.control_q = mp.Queue(n_save_proc*2)
        self.save_proc_group = []
        for n in range(n_save_proc):
            proc = mp.Process(target=save_img_proc, args=(self.control_q,), daemon=True)
            proc.start()
            self.save_proc_group.append(proc)

        self.mem_list = []
        self.n_memory_slot = 20
        max_memory_size = 40*3072*1536
        for n in range(self.n_memory_slot):
            shm = shared_memory.SharedMemory(create=True, size=max_memory_size)
            self.mem_list.append(shm)

        self.shm_idx = 0

    def imwrite(self, fn, img):
        this_shm = self.mem_list[self.shm_idx]
        self.shm_idx += 1
        if self.shm_idx == len(self.mem_list):
            self.shm_idx = 0
        numpy_view = np.ndarray(img.shape, dtype=img.dtype, buffer=this_shm.buf)
        numpy_view[:] = img[:]
        self.control_q.put([this_shm.name, img.shape, fn] )

    def close(self):
        while not self.control_q.empty():
            time.sleep(0.1)
        time.sleep(1)
        for item in self.save_proc_group:
            item.terminate()

