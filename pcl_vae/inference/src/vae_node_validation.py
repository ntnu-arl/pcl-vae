import os
import time
import math
import torch
import argparse
import itertools
import numpy as np
import yaml
import cv2
from cv_bridge import CvBridge

# Dataset Libraries
import pcl_vae
from pcl_vae.datasets.range_image_dataset import RangeImageDataset
from torch.utils.data import DataLoader

# VAE Libraries
from pcl_vae.inference.scripts.VAENetworkInterfaceValidation import VAENetworkInterfaceValidation
from pcl_vae.networks.Loss.loss_functions import *

# Paths
BASE_PATH = pcl_vae.inference.__path__[0]
BASE_DATASET_PATH = pcl_vae.datasets.__path__[0]

# Use argparser to distinguish between robot type
parser = argparse.ArgumentParser()
parser.add_argument('--robot_type', default="ground")


class PCLVAEValidation():
    def __init__(self):

        # Load params
        self.load_params()

        # Load the testing dataset
        DATASET_PATH = os.path.join(BASE_DATASET_PATH, self.base_dataset_path, self.robot_type, self.set_name)
        testing_dataset = RangeImageDataset(root_dir=DATASET_PATH)
        print(f"[DATASET]: Path to Dataset loaded: {DATASET_PATH}")
        print(f"[DATASET]: Total number of range images loaded: {len(testing_dataset)}")

        # Create DataLoader for the testing dataset
        test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

        # Display the range image from the training set
        # self.display_dataset(test_loader)

        # Network Initialization
        self.net_interface = VAENetworkInterfaceValidation(self.robot_type, self.model_name, self.latent_space, self.device)
        self.test_loader = test_loader
        self.bridge = CvBridge()
        
        # Initialize MSE accumulation variables
        self.loss_sum = 0.0
        self.num_images = 0

        # Model Validation
        self.model_validation(self.test_loader) 

    def load_params(self):
        # Get the robot name
        args = parser.parse_args()
        if args.robot_type:
            self.robot_type = args.robot_type
            print("[ROBOT]: Robot name: ", self.robot_type)

        # Load config file
        CONFIG_PATH = os.path.join(BASE_PATH, "config", self.robot_type, "vae_validation_config.yaml")
        with open(CONFIG_PATH, "r") as file:
            data = yaml.safe_load(file)

        self.device = data['device']
        self.latent_space = data['latent_space']
        self.image_height = data['image_height']
        self.image_width = data['image_width']
        self.image_max_depth = data['image_max_depth']
        self.image_min_depth = data['image_min_depth']
        self.invalid_pixel_value = data['invalid_pixel_value']
        self.voxel_size = data['voxel_size']
        self.calculate_loss = data['calculate_loss']
        self.current_index = data['current_index']
        self.base_dataset_path = data['base_dataset_path']
        self.set_name = data['set_name']
        self.h_fov = data['h_fov']
        self.v_fov = data['v_fov']
        self.model_name = data['model_name']

    def model_validation(self, test_dataset_loader):
    
        self.load_next_image = True

        # Initialize an iterator for the dataset
        test_dataset_iterator = iter(test_dataset_loader)

        # Skip the x images 
        test_dataset_iterator = itertools.islice(test_dataset_iterator, self.current_index, None)

        while self.current_index < len(test_dataset_loader):
            if self.load_next_image:
                print(f"[DATASET]: Loaded range images: {self.current_index}/{len(test_dataset_loader)}")
                
                # Load the next image from the dataset
                data = next(test_dataset_iterator)

                # Send data on GPU
                data = data.to(self.device)
                
                # Process input data
                processed_image, image_to_reconstruct = self.process_for_validation(data)

                # Range image reconstruction
                reconstruction_normalized, means, logvar, compute_time = self.net_interface.forward(image_to_reconstruct.cpu().numpy())
                reconstruction = reconstruction_normalized * self.image_max_depth

                # Loss calculation
                if self.calculate_loss:
                    loss, reconstruction_loss, kld_loss, voxelized_range_image = occupancy_loss(processed_image, reconstruction_normalized, means, logvar, self.latent_space, self.voxel_size, self.image_max_depth, self.image_min_depth, self.h_fov, self.v_fov)
                    self.loss_sum += kld_loss       #loss, reconstruction_loss, kld_loss
                    self.num_images += 1
                    print(self.num_images, ") loss:", loss.item(), "reconstruction_loss: ", reconstruction_loss.item(), "kld_loss: ", kld_loss.item())
                else:
                    loss, reconstruction_loss, kld_loss, voxelized_range_image = occupancy_loss(image_to_reconstruct, reconstruction_normalized, means, logvar, self.latent_space, self.voxel_size, self.image_max_depth, self.image_min_depth, self.h_fov, self.v_fov)
                    self.display_range_images_for_comparison(data, voxelized_range_image, reconstruction_normalized)
                
                # Set flag to false to wait for the next key press to load the next image
                self.load_next_image = False

            if self.calculate_loss:
                self.load_next_image = True
                self.current_index += 1
                cv2.destroyAllWindows()
            else:
                # Check for user input to load the next image
                key = cv2.waitKey(10)
                if key == ord('n'):  # 'n' for next image
                    self.load_next_image = True
                    self.current_index += 1
                    cv2.destroyAllWindows()

        # Calculate the average MSE after all images are processed
        if self.num_images > 0:
            avg_loss = self.loss_sum / self.num_images
            print(f"Average MSE Loss over {self.num_images} range images: {avg_loss:.3f}")

    def process_for_validation(self, org_img):

        # Normalize and process the range image
        processed_image = org_img.clone()
        processed_image[processed_image > self.image_max_depth] = self.image_max_depth
        processed_image[processed_image < self.image_min_depth] = self.invalid_pixel_value
        processed_image = processed_image / self.image_max_depth
        processed_image[processed_image < 0] = self.invalid_pixel_value 

        # Fill
        processed_image_np = processed_image.cpu().numpy().squeeze(0).squeeze(0)
        invalid_mask = (processed_image == self.invalid_pixel_value).to(torch.uint8)
        invalid_mask_np = invalid_mask.cpu().numpy().squeeze(0).squeeze(0)

        filled_image_np = cv2.inpaint(processed_image_np, invalid_mask_np, inpaintRadius=3, flags=cv2.INPAINT_NS) 
        filled_image = torch.from_numpy(filled_image_np).float().unsqueeze(0).unsqueeze(0)
        filled_image = filled_image.to(self.device)   
                                
        range_image_to_reconstruct = filled_image.clone()
        range_image_to_reconstruct[range_image_to_reconstruct < 0] = -2*self.invalid_pixel_value 

        return processed_image, range_image_to_reconstruct
    
    def display_range_images_for_comparison(self, *range_images):
        """
        Display multiple range images for comparison. This function takes an arbitrary
        number of images as arguments and displays them in a single concatenated window.
        
        Args:
            range_images: Variable number of range image arguments (original, reconstructed, masks, etc.).
        """
        # Prepare a list to store the processed images
        processed_images = []

        for image in range_images:
            # Convert to NumPy, detach, and move to CPU if necessary
            numpy_array = image.detach().cpu().numpy() if hasattr(image, 'detach') else image
            numpy_array = numpy_array[0, 0, :, :]  # Assuming shape [1, 1, 64, 512]

            # Normalize the array to 0-255 if necessary
            numpy_array = cv2.normalize(numpy_array, None, 0, 255, cv2.NORM_MINMAX)

            # Convert to unsigned 8-bit integer
            numpy_array = numpy_array.astype(np.uint8)

            # Append the processed image to the list
            processed_images.append(numpy_array)

        # Vertically concatenate all processed images
        Vert = np.concatenate(processed_images, axis=0)
        
        # Display the concatenated image using OpenCV
        cv2.imshow('Range Images Comparison', Vert)
        # cv2.waitKey(0)  # Wait for a key press to close the window
        # cv2.destroyAllWindows()

    def display_dataset(self, test_loader):
        '''
        Function to show range images from the dataset
        '''
        current_index = 0
        load_next_image = True

        # Initialize an iterator for the dataset
        test_dataset_iterator = iter(test_loader)

        while current_index < len(test_loader):
            if load_next_image:
                data = next(test_dataset_iterator)
                image_tensor = data[0]
                numpy_array = image_tensor.detach().cpu().numpy()
                numpy_array = numpy_array.squeeze()

                # Normalize the array to 0-255 if necessary
                numpy_array = cv2.normalize(numpy_array, None, 0, 255, cv2.NORM_MINMAX)

                # Convert to unsigned 8-bit integer
                numpy_array = numpy_array.astype(np.uint8)

                # Display the image using OpenCV
                cv2.imshow('Range Image', numpy_array)
                load_next_image = False

            key = cv2.waitKey(10)
            if key == ord('n'):  # 'n' for next image
                load_next_image = True
                current_index += 1
                cv2.destroyAllWindows()

if __name__ == "__main__":
    PCLVAEValidation()

