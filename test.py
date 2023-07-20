import os

folder_name = r"train"  # Use a raw string by prefixing "r" before the string

# Get the absolute path of the folder
folder_path = os.path.abspath(folder_name)
file_path = r'C:\Users\User\Desktop\github\tuftsProject\train_v2\train'
img_path = r'C:\Users\User\Desktop\github\tuftsProject\train_v2\train\TRAIN_00001.jpg'


print("Absolute path of the folder:", folder_path)


if os.path.exists(img_path):
    print("File exists and can be accessed.")
else:
    print("File does not exist or cannot be accessed.")
