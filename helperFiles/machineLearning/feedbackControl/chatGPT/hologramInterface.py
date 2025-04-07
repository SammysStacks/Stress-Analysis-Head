import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import shutil

class PNGHandler(FileSystemEventHandler):
    def __init__(self, dropbox_folder):
        self.dropbox_folder = dropbox_folder

    def on_created(self, event):
        print(f"File created: {event.src_path}")
        if event.src_path.endswith(".png"):
            print(f"New PNG detected: {event.src_path}")
            time.sleep(1)  # Wait to ensure the file is fully created

            if os.path.exists(event.src_path):
                self.save_to_dropbox(event.src_path)
            else:
                print(f"File not found: {event.src_path}")

    def save_to_dropbox(self, file_path):
        try:
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(self.dropbox_folder, file_name)

            # Copy the file to the Dropbox folder
            shutil.copy2(file_path, destination_path)
            print(f"File {file_path} copied to Dropbox folder: {destination_path}")
        except Exception as e:
            print(f"Error saving file to Dropbox: {e}")

if __name__ == "__main__":
    # Specify the folder to watch
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_to_watch = os.path.join(current_dir, 'therapyHelperFiles', '_savedImages')
    dropbox_folder = 'DropBox Folder Path'

    if not os.path.exists(folder_to_watch):
        os.makedirs(folder_to_watch)

    if not os.path.exists(dropbox_folder):
        os.makedirs(dropbox_folder)

    print(f"Monitoring folder: {folder_to_watch} for PNG uploads...")

    # Set up the observer
    event_handler = PNGHandler(dropbox_folder)
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()
