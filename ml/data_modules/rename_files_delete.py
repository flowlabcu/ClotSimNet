import os

# Path to the directory containing the files
directory_path = '/mnt/d/clotsimnet_data/clotsimnet_data_test_updated/cnn_data_crop'

# List all files in the directory
files = os.listdir(directory_path)

# Loop through each file in the directory
for filename in files:
    # Check if '_no_crop_cnn' is in the filename
    if '_no_crop_cnn' in filename:
        # Create the new file name by replacing '_no_crop_cnn' with '_crop'
        new_filename = filename.replace('_no_crop_cnn', '_crop')
        
        # Create full paths for old and new filenames
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)

        print(f'Renamed: {filename} -> {new_filename}')
