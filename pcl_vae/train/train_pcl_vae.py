import os
import time
import yaml
import argparse

# Dataset Libraries
from pcl_vae.datasets.range_image_dataset import RangeImageDataset
from torch.utils.data import DataLoader

# VAE Libraries
import pcl_vae.train
from pcl_vae.networks.VAE.vae import *
from pcl_vae.networks.Loss.loss_functions import *
from pcl_vae.networks.Loss.running_loss import RunningLoss
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# Path
BASE_TRAIN_PATH = pcl_vae.train.__path__[0]
BASE_DATASET_PATH = pcl_vae.datasets.__path__[0]

# Get the robot name
parser = argparse.ArgumentParser() 
parser.add_argument('--robot_type', default="ground")


class PCLVAETrain():
    def __init__(self):

        # Load params
        self.load_params()
    
        # Load the dataset
        DATASET_PATH = os.path.join(BASE_DATASET_PATH, self.base_dataset_path, self.robot_type, self.set_name)
        train_dataset = RangeImageDataset(root_dir=DATASET_PATH)
        print(f"[DATASET]: Path to Dataset loaded: {DATASET_PATH}")
        print(f"[DATASET]: Total number of range images loaded: {len(train_dataset)}")

        # Create DataLoader for the training set
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Display the range image from the training set
        # self.display_dataset(train_loader)

        # Model
        if self.robot_type == "aerial":
            model = VAE(input_channels=1, latent_dim=self.latent_space)
        elif self.robot_type == "ground":
            model = VAE_2(input_channels=1, latent_dim=self.latent_space)
        else:
            print("[ROBOT]: Unknown robot name for loading the VAE")
            return
        model = model.to(self.device)
        summary(model, (1, self.image_height, self.image_width))
        
        # Loss Function
        loss = occupancy_loss

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # TensorBoard
        self.experiment_name = self.robot_type + "_model"
        self.experiment_path = os.path.join(BASE_TRAIN_PATH, "weights", self.robot_type, self.experiment_name)
        TENSOR_BOARD_PATH = os.path.join(self.experiment_path, "tensorboard", f"{self.experiment_name}_LD_{self.latent_space}_epoch_{self.num_epochs}_batch_{self.batch_size}_range_{int(self.image_max_range)}_voxel_{int(self.voxel_size*100)}")
        writer = SummaryWriter(log_dir=TENSOR_BOARD_PATH)

        # Train the model
        model = self.train_model(train_loader, model, loss, optimizer, writer)

    def load_params(self):
        # Get the robot name
        args = parser.parse_args()
        if args.robot_type:
            self.robot_type = args.robot_type
            print("[ROBOT]: Robot name: ", self.robot_type)

        # Load config file
        CONFIG_PATH = os.path.join(BASE_TRAIN_PATH, "config", self.robot_type, "train_config.yaml")
        with open(CONFIG_PATH, "r") as file:
            data = yaml.safe_load(file)
            print(data)

        self.device = data['device']
        self.base_dataset_path = data['base_dataset_path']
        self.set_name = data['set_name']
        self.latent_space = data['latent_space']
        self.num_epochs = data['num_epochs']
        self.batch_size = data['batch_size']
        self.learning_rate = data['learning_rate']
        self.h_fov = data['h_fov']
        self.v_fov = data['v_fov']
        self.image_height = data['image_height']
        self.image_width = data['image_width']
        self.image_min_range = data['image_min_range']
        self.image_max_range = data['image_max_range']
        self.invalid_pixel_value = data['invalid_pixel_value']
        self.voxel_size = data['voxel_size']
       
    def train_model(self, train_dataset_loader, model, loss_fn, optimizer, writer):
        '''
        Function to train the given input model based on the given data
        ----------------------------------------------------------------
        Dimensions of Variables
            - data                      :   torch.Size([B, C, H, W])
            - processed_image           :   torch.Size([B, C, H, W])
            - range_data_to_reconstruct :   torch.Size([B, C, H, W])
            - reconstructed_image       :   torch.Size([B, C, H, W])
            - means                     :   torch.Size([B, L])
            - log_vars                  :   torch.Size([B, L])
        ----------------------------------------------------------------
        '''

        # Initialize the training time counter
        training_start_time = time.time()

        # Anomaly detection (e.g., NaNs, Infs in the gradients)
        torch.autograd.set_detect_anomaly(True)
        
        # Initialize the loss and the optimizer
        loss_meter = RunningLoss(self.batch_size)
        
        # Initialize the number of batches
        num_batches = len(train_dataset_loader)

        # Initialize the number of epochs
        for epoch in range(self.num_epochs):

            model.train()
            
            # Initialize the epoch time counter
            epoch_start_time = time.time()
            
            # Initialize the number of batches
            for batch_idx, data in enumerate(train_dataset_loader):

                # Initialize the batch time counter
                batch_start_time = time.time()

                # Zeroing gradients
                optimizer.zero_grad()

                # Send data on GPU
                data = data.to(self.device)

                # Normalized the input data
                processed_image, image_to_reconstruct = self.process_for_training(data)

                # Forward pass
                reconstructed_image, means, log_vars  = model(image_to_reconstruct)
            
                # Loss calculation
                loss, reconstruction_loss, kld_loss, voxelized_range_image = loss_fn(processed_image, reconstructed_image, means, log_vars, self.latent_space, self.voxel_size, self.image_max_range, self.image_min_range, self.h_fov, self.v_fov)
                
                # Update the loss meter
                loss_meter.update(loss.item())
                avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update
                if batch_idx % 10 == 0 and batch_idx != 0:
                    print('Train/Loss', loss.item()/self.batch_size)
                    print('Train/Rec Loss', reconstruction_loss.item()/self.batch_size)
                    print('Train/KLD Loss', kld_loss.item()/self.batch_size)
                    # print('Train/Edge Loss', edge_loss.item()/BATCH_SIZE)
                    print(f"[TRAINING] Epoch: {epoch}/{self.num_epochs} Batch: {batch_idx}/{num_batches} Avg. Train Loss: {loss.item()/self.batch_size:.4f}, KL Div Loss.: {kld_loss.item()/self.batch_size:.4f}"\
                        f" Time: {time.time() - batch_start_time:.2f}s, Est. time remaining: {(num_batches - batch_idx)*avg_iter_time :.2f}s")

                # TensorBoard
                writer.add_scalar('Train/Loss', loss.item()/self.batch_size, epoch * num_batches + batch_idx)
                writer.add_scalar('Train/Rec Loss', reconstruction_loss.item()/self.batch_size, epoch * num_batches + batch_idx)
                writer.add_scalar('Train/KLD Loss', kld_loss.item()/self.batch_size, epoch * num_batches + batch_idx)
                # writer.add_scalar('Train/Edge Loss', edge_loss.item()/BATCH_SIZE, epoch * num_batches + batch_idx)
            
            # Print the statistics
            print('Epoch: %d, Loss: %.4f, Time: %.4f' %(epoch, loss_meter.avg, time.time() - epoch_start_time))
            
            # Reset the loss meter
            loss_meter.reset()

        writer.flush()

        # Save the model
        print("[MODEL]: Saving model...")
        torch.save(model.state_dict(), os.path.join(self.experiment_path, '%s_LD_%d_epoch_%d_batch_%d_range_%d_voxel_%d.pth' % (self.experiment_name, self.latent_space, self.num_epochs, self.batch_size, self.image_max_range, self.voxel_size*100)))
        print("[MODEL]: Model saved at ", self.experiment_path)

        # Record the training end time
        training_end_time = time.time()

        # Calculate the training duration in seconds and convert to minutes/hours if necessary
        duration = training_end_time - training_start_time
        print(f"[STATISTICS]: Total training duration: {duration:.2f} seconds")
        print(f"[STATISTICS]: Total training duration: {duration / 60:.2f} minutes")

        return model

    def process_for_training(self, input_image):
        '''
        Function to process the input range image for training
        ------------------------------------------------------------------------
        Parameter 
            - processed_input_image :       Valid pixels: [0.027, 1.0] 
                                            Invalid pixels: -1
            - range_image_to_reconstruct :  Valid pixels: [0.027, 1.0] 
                                            Invalid pixels: 2
        ------------------------------------------------------------------------
        '''

        # Creates a deep copy
        processed_input_image = input_image.clone()
        # Clamping values above MAX_RANGE
        processed_input_image[processed_input_image > self.image_max_range] = self.image_max_range 
        # Handles invalid data points
        processed_input_image[processed_input_image < self.image_min_range] = self.invalid_pixel_value
        # Data normalization
        processed_input_image = processed_input_image / self.image_max_range
        # Maintaning the invalid data points 
        processed_input_image[processed_input_image < 0] = self.invalid_pixel_value

        range_image_to_reconstruct = processed_input_image.clone()
        range_image_to_reconstruct[range_image_to_reconstruct < 0] = -2*self.invalid_pixel_value

        return processed_input_image, range_image_to_reconstruct
    
    def display_dataset(self, train_loader):
        '''
        Function to show range images from the dataset
        '''
        current_index = 0
        load_next_image = True

        # Initialize an iterator for the dataset
        test_dataset_iterator = iter(train_loader)

        while current_index < len(train_loader):
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
    PCLVAETrain()