import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
import cv2

data_path=Path("data/raw")



def check_missing_files(dir1,dir2):
    
    # comparing names of files in two directories
    files1=set(os.listdir(dir2))
    files2=set(os.listdir(dir1))

    for file in files1:
        if file.endswith(".txt"):
            label_file=file.rsplit(".",1)[0]+".jpg"
            if label_file not in files2:
                print(f"Missing label file for {file} in {dir2}")


def check_annotations_of_labels(label_dir):

    for labels_file in os.listdir(label_dir):
        if labels_file.endswith(".txt"):
            with open(label_dir/labels_file,"r") as f:
                lines=f.readlines()
                for line in lines:
                    parts=line.strip().split()
                    if len(parts)!=5:
                        print(f"Invalid annotation format in {labels_file}: {line}")
                    else:
                        class_id,x_center,y_center,width,height=parts
                        try:
                            class_id=int(class_id)
                            x_center=float(x_center)
                            y_center=float(y_center)
                            width=float(width)
                            height=float(height)
                        except ValueError:
                            print(f"Invalid annotation values in {labels_file}: {line}")



def transform_images(image_dir):
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg"):
            image_path=image_dir/image_file
            image=cv2.imread(str(image_path))
            if image is not None:
                resized_image=cv2.resize(image,(640,640))
                cv2.imwrite(str(image_path),resized_image)
            else:
                print(f"Could not read image {image_file} in {image_dir}")

def preprocess_data_for_yolo(data_path):
    
    image_dir=data_path/"images"
    label_dir=data_path/"labels"

    image_train_dir=image_dir/"train"
    label_train_dir=label_dir/"train"
    image_val_dir=image_dir/"val"
    label_val_dir=label_dir/"val"

    check_missing_files(image_train_dir,label_train_dir)
    check_missing_files(image_val_dir,label_val_dir)
    check_annotations_of_labels(label_train_dir)
    check_annotations_of_labels(label_val_dir)

    transform_images(image_train_dir)
    transform_images(image_val_dir)




def main():
    preprocess_data_for_yolo(data_path)


if __name__=="__main__":    main()


    