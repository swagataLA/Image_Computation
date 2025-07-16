import os
import shutil

# Directory containing your images
source_dir = "C:/Users/swaga/Downloads/hand_drawn_dataset-20240507T203228Z-001/hand_drawn_dataset"
# Directory where sorted images will be placed
destination_dir = "c:/Users/swaga/Documents/CS510/PA1/cs510tutorials (1)/cs510tutorials/PA3"

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate through all the files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.png'):
        # Extract the label from the filename (e.g., "5_example.png" gives us '5')
        label = filename.split('_')[0]
        
        # Create a directory for the label if it doesn't already exist
        label_dir = os.path.join(destination_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Move the file into the corresponding label directory
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(label_dir, filename)
        shutil.move(src_path, dest_path)

print("Images organized by labels successfully.")