'''
File description: Delete all files that end in :Zone.Identifier when copying files from Windows to WSL.
'''

import os
import glob

def remove_zone_identifier_files(path: str):
    for root, dirs, files in os.walk(path):
        for file in glob.glob(os.path.join(root, '*:Zone.Identifier')):
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")

# Example usage
# remove_zone_identifier_files('/path/to/directory')

path = '/home/josh/'

remove_zone_identifier_files(path=path)