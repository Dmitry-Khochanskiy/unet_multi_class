#!/usr/bin/env python
#coding: utf-8
import argparse
import sys
import inference
import os
''' script for single image or batch predicition. Accepts folder or single file path'''
def accept_arguments():
    script_path = sys.path[0]
    my_parser = argparse.ArgumentParser(description='Inference of single image or a folder')
    my_parser.add_argument('--image_path',
                       type=str,
                       help='Path to single image or folder')

    my_parser.add_argument('--model_path',
                       type=str,
                       help='Path to trained model')

    my_parser.add_argument('--masked_img_path',
                       type=str,
                       help='Path for masked images to be saved')

    my_parser.add_argument('--masks_path',
                       type=str,
                       help='Path for maskes to be saved')

    my_parser.add_argument('--show_results',
                       action='store_true',
                       default=False,
                       help='Showing resulting figure for single image inference')

    args = my_parser.parse_args()

    img_path = args.image_path
    model_path = args.model_path
    masked_img_path = args.masked_img_path
    masks_path = args.masks_path
    show_results =  args.show_results
    if os.path.isfile(img_path):
        model, parameters = load_model(model_path)
        prediction = inference.show_img_with_seg(image_path, model, parameters[0],  parameters[1], show_results)
    else:
        model, parameters = load_model(model_path)
        inference.batch_prediction(images_folder_path, model, parameters[0],  parameters[2], mask_save_path)


if __name__ == '__main__':
    accept_arguments()
