import os
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import shutil
from typing import Sequence

# Define a palette
PALETTE = [0]*15
PALETTE[0:3] = [255, 0, 0]
PALETTE[3:6] = [0,255,0]
PALETTE[6:9] = [0,0,255]
PALETTE[9:12] = [153, 0, 255]
PALETTE[12:15] = [255, 255, 255]

def format_and_save_mask(mask_im:np.ndarray,
                palette:Sequence,
                output_filepath:str) -> str:
    if os.path.exists(output_filepath):
        return output_filepath
    else:
        mask_image = Image.fromarray(mask_im, mode='P')
        mask_image.putpalette(palette)
        mask_image.save(output_filepath)
        return output_filepath

def create_mask_from_segmentation(segmentation_filepath:str) -> np.ndarray:
    # Open the segmentation image and convert it to a NumPy array
    with Image.open(segmentation_filepath) as seg_image:
        seg_im = np.asarray(seg_image)
    
    # Create a binary mask from the segmentation image
    mask_im = seg_im.copy()
    mask_im[mask_im > 0] = 1  # Set all non-zero values to 1
    
    return mask_im

def move_and_convert(jpg_filepath:str, dest_directory:str,img_label:str):
    # Ensure the destination directory exists
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # Get the filename from the source path and construct the destination path for the JPG
    filename = os.path.basename(jpg_filepath)
    # Define the new PNG filename and its full path
    png_filename = os.path.splitext(filename)[0] + img_label + '.png'
    dest_png_path = os.path.join(dest_directory, png_filename)
    if os.path.exists(dest_png_path):
        return dest_png_path
    
    # Open the moved JPG image and convert it to PNG
    with Image.open(jpg_filepath) as img:
        img.save(dest_png_path, 'PNG')

    return dest_png_path

def process_train_directory(directory):
    normal_dir = os.path.normpath(os.path.join(directory, 'normal'))
    tumor_dir = os.path.normpath(os.path.join(directory, 'tumor'))
    seg_dir = os.path.normpath(os.path.join(directory, 'seg'))
    img_dir = os.path.normpath(os.path.join(os.path.dirname(directory), 'train'))
    saved_impaths = []

    normal_filepaths = [os.path.join(normal_dir, filename) for filename in os.listdir(normal_dir) if filename.endswith('.jpg')]
    normal_label = '[1000]'
    for normal_filepath in tqdm(normal_filepaths, desc='Processing normal images'):
        saved_impaths.append(move_and_convert(normal_filepath, img_dir, normal_label))

    tumor_filepaths = [os.path.join(tumor_dir, filename) for filename in os.listdir(tumor_dir) if filename.endswith('.jpg')]
    tumor_label = '[1100]'
    for tumor_filepath in tqdm(tumor_filepaths, desc='Processing tumor images'):
        saved_impaths.append(move_and_convert(tumor_filepath, img_dir, tumor_label))

    num_input_ims = len(normal_filepaths) + len(tumor_filepaths)
    if num_input_ims != len(saved_impaths):
        raise ValueError("Number of saved images does not match the number of input images.")
    else:
        shutil.rmtree(directory)
    return

def process_val_directory(directory):
    normal_dir = os.path.normpath(os.path.join(directory, 'normal'))
    tumor_dir = os.path.normpath(os.path.join(directory, 'tumor'))
    seg_dir = os.path.normpath(os.path.join(directory, 'seg'))
    img_dir = os.path.normpath(os.path.join(directory, 'img'))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    mask_dir = os.path.normpath(os.path.join(directory, 'mask'))
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    saved_im_paths = []
    normal_filepaths = [os.path.join(normal_dir, filename) for filename in os.listdir(normal_dir) if filename.endswith('.jpg')]
    normal_label = ''
    for normal_filepath in tqdm(normal_filepaths, desc='Processing normal images'):
        saved_im_paths.append(move_and_convert(normal_filepath, img_dir, normal_label))
    tumor_filepaths = [os.path.join(tumor_dir, filename) for filename in os.listdir(tumor_dir) if filename.endswith('.jpg')]
    tumor_label = ''
    for tumor_filepath in tqdm(tumor_filepaths, desc='Processing tumor images'):
        saved_im_paths.append(move_and_convert(tumor_filepath, img_dir, tumor_label))
    saved_mask_paths = []
    normal_imsize = Image.open(normal_filepaths[0]).size
    normal_mask = np.zeros((normal_imsize[1], normal_imsize[0]), dtype=np.uint8)
    for normal_filepath in tqdm(normal_filepaths,desc='Processing normal masks'):
        filename = os.path.basename(normal_filepath).replace('.jpg', '.png')
        mask_filepath = os.path.join(mask_dir, filename)
        saved_mask_paths.append(format_and_save_mask(normal_mask, PALETTE, mask_filepath))

    seg_filepaths = [os.path.join(seg_dir, filename) for filename in os.listdir(seg_dir) if filename.endswith('.png')]
    for seg_filepath in tqdm(seg_filepaths, desc='Processing tumor masks'):
        mask_im = create_mask_from_segmentation(seg_filepath)
        filename = os.path.basename(seg_filepath)
        mask_filepath = os.path.join(mask_dir, filename)
        saved_mask_paths.append(format_and_save_mask(mask_im, PALETTE, mask_filepath))
    
    if len(normal_filepaths) + len(tumor_filepaths) != len(saved_im_paths):
        raise ValueError("Number of saved images does not match the number of input images.")
    # delete everything in the normal and tumor directories using shutil.rmtree
    shutil.rmtree(normal_dir)
    shutil.rmtree(tumor_dir)

    if len(normal_filepaths) + len(seg_filepaths) != len(saved_mask_paths):
        raise ValueError("Number of saved masks does not match the number of input masks.")
    # delete everything in the seg directory using shutil.rmtree
    shutil.rmtree(seg_dir)

    # rename the directory from 'validate' to 'val'
    os.rename(directory, directory.replace('validate', 'val'))
    return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/t2f', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist.")

    print("Running on Train Subset")
    train_path = os.path.join(args.data_dir, 'training')

    process_train_directory(train_path)
    val_path = os.path.join(args.data_dir, 'validate')

    print("Running on Validate Subset")
    process_val_directory(val_path)
