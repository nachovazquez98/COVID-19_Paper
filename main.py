import os
import psutil
from multiprocessing import Pool
from subprocess import check_call
'''
https://stackabuse.com/executing-shell-commands-with-python/
https://unix.stackexchange.com/questions/238180/execute-shell-commands-in-python
https://www.circuitbasics.com/run-linux-commands-with-python/
'''
def install_pip():
    os.system('sudo apt-get install software-properties-common')
    os.system('sudo apt-add-repository universe')
    os.system('sudo apt-get update')
    os.system('sudo apt-get install python3-pip')
    os.system('python3 -m pip install --upgrade --force-reinstall pip')

def install_pipenv():
    os.system('pip3 install pipenv')
    os.system('sudo /home/ubuntu/.local/bin/easy_install pipenv')
    os.system('pipenv install')
    os.system('pipenv shell')

commands = [
    python covid_train_ml.py & python sys_monitor.py
]

def sample(command):
    check_call(command, shell=True)


if __name__ == '__main__':
    #install_pip()
    #install_pipenv()
    with Pool(len(commands)) as p:
        p.map(sample, commands)