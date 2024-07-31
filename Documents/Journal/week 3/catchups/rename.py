import os
import re

def rename_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            new_filename = re.sub(r'[^\w\-_\.]', '_', filename)  # Replace special characters with underscores
            new_filename = new_filename.replace(' ', '_')  # Replace spaces with underscores
            old_file = os.path.join(root, filename)
            new_file = os.path.join(root, new_filename)
            if old_file != new_file:
                print(f'Renaming: {old_file} -> {new_file}')
                os.rename(old_file, new_file)

if __name__ == "__main__":
    directory = "/home/dadudekc/project/TradingRobotPlug/Documents/Journal/week 3/catchups"  # Specify your directory here
    rename_files(directory)
