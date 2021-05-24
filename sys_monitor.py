import csv  
from time import time, sleep
from datetime import datetime
import cpuinfo
import psutil
import pathlib
import os

def get_fields():
    hour = str(datetime.now())[11:19]
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq()[0]
    thread_percent = psutil.cpu_percent(interval=1, percpu=True)
    ram_percent = psutil.virtual_memory().percent
    row =  [hour, cpu_percent, cpu_freq, ram_percent, thread_percent]
    return row

def log_name():
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    cpu_name = cpu_name.replace(" ", "_")
    n_cores = psutil.cpu_count(logical=False)
    n_threads = psutil.cpu_count()
    date = str(datetime.now())[:10]
    hour = str(datetime.now())[11:16]
    file_name = str(cpu_name) + "_" + str(n_cores) + "cores_" + str(n_threads) + "threads_" + date + "_" + hour
    return file_name

if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute()
    os.chdir(path)

    file_name = log_name()
    column_names = ["hour", "cpu_percent", "cpu_freq", "ram_percent", "thread_percent"]
    with open(str(file_name)+".csv", 'a') as f:
        writer1 = csv.writer(f)
        writer1.writerow(column_names)
        while True:
            with open(str(file_name)+".csv", 'a') as ff:
                data_columns = get_fields()
                writer2 = csv.writer(ff)
                writer2.writerow(data_columns)
                sleep(29 - time() % 29)
