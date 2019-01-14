import requests
import json
import argparse
import psutil
import time
import configparser
import os
import subprocess

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", help="config path")

args = vars(ap.parse_args())

# telegram chat id 
# find IDBot in telegram application to get your id
chat_id = "504489643"

# global process ident
pid = 0
# if terminated send message once
terminated = False

def read_config(path):
    """Read config from path
    Args:
    :path : path to config file

    :return : dict of config values
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config


class Handler(FileSystemEventHandler):
    """Watchdog event handler"""
    def on_created(self, event):
        print(event)

    def on_deleted(self, event):
        print(event)

    def on_moved(self, event):
        print(event)

    def on_modified(self, event):
        global pid
        global terminated
        print(event)
        pid_t = int(read_config(args['config'])["TRAIN"]["PID"])
        if pid_t != pid:
            print("PID changed: %s" % pid_t)
            pid = pid_t
            terminated = False

def on_terminate(pid, chat_id, descr):
    # send message via telegram bor when process terminated
    data = {"message":
                    {
                    "chat_id": chat_id,
                    "text": "text",
                    "pid": pid,
                    "descr": descr
                    }
            }

    r = requests.get("https://favorner.pythonanywhere.com/bot779957072:AAFauKaDDabwb77W6UCogI3S3kT6Jud16y0", data=json.dumps(data))
    # print(r)
    return r

def main():
    global pid
    global terminated
    config = read_config(args['config'])
    pid = int(config["TRAIN"]["PID"])

    # create watchdog file change handler
    observer = Observer()
    observer.schedule(Handler(), path=args['config'].split(os.sep)[0], recursive=True)
    observer.start()

    while True:
        # get process status
        status = psutil.pid_exists(pid)
        # if process status is False send message to `chat_id`
        if not status and not terminated:
            print("---------------------------------")
            print("Process terminated")
            on_terminate(pid, chat_id, config["TRAIN"]["Description"])
            # print("Trying to start new %s process" % config["TRAIN"]["Description"])
            terminated = True

            # TODO:
            # start new train process if previous terminated
            # get exit code
            # create subprocess
            # 

            # q = input("Continue with new PID? [Y/n]")
            # if q != "n":
            #   new_pid = int(input("Enter new PID: "))
            #   pid = new_pid
            # else:
            #   break
        
        time.sleep(2)

if __name__ == "__main__":
    main()