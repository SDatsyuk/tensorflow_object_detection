from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import time



def watch(path):
	observer = Observer()
	observer.schedule(Handler(), path=path, recursive=True)
	observer.start()


if __name__ == "__main__":
	watch("pid")

	try:
	    while True:
	        time.sleep(1)
	except KeyboardInterrupt:
	    observer.stop()
	observer.join()