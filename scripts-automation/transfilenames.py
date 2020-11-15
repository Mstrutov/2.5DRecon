import os
import re
import numpy as np

# getting the list of files
work_dir = r'datasets\\videos'
files = os.listdir(work_dir)
files = np.array(files)

# selecting '...(<number>).mp4' files
mask = [re.search(r'.*(\)\.mp4)', file) is not None for file in files]
masked_files = files[mask]

for file_name in masked_files:
    # generating a new file name using a corresponding number (here by incrementing it)
    new_file_name = re.sub(r'.*\((.*)\)', lambda x: str(int(x.group(1)) + 1), file_name)

    # renaming selected files
    os.rename(work_dir + r'\\' + file_name, work_dir + r'\\' + new_file_name)
