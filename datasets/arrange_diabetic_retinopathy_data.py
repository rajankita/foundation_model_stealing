import os
import csv
import shutil


data_root = '/home/ankita/scratch/Datasets/Diabetic_Retinopathy/train'
data_out_dir = '/home/ankita/scratch/Datasets/Diabetic_Retinopathy/training_imgs'
label_file = '/home/ankita/scratch/Datasets/Diabetic_Retinopathy/trainLabels.csv'


# Prepare output directory and sub-directories for classes
os.makedirs(data_out_dir, exist_ok=True)

# Read labels from csv file
with open(label_file) as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        filename, class_id = row
        filename = filename + '.jpeg'
        print(filename, class_id)
        
        img_src = os.path.join(data_root, filename)
        img_dest_dir = os.path.join(data_out_dir, class_id)
        if not os.path.exists(img_dest_dir):
            os.makedirs(img_dest_dir)
        
        shutil.copy(img_src, os.path.join(img_dest_dir, filename))
        # break

