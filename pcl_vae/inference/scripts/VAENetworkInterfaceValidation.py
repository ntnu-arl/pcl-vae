import time
import os
import torch
import pcl_vae
from pcl_vae.networks.VAE.vae import *

# Weights Path
BASE_PATH = pcl_vae.__path__[0]

class VAENetworkInterfaceValidation():
    def __init__(self, robot_type, model_name, latent_space_dim, device):
        self.robot_type = robot_type
        self.model_name = model_name
        self.latent_space = latent_space_dim
        self.device = device

        # Load networks in constructor
        try:
            if robot_type == "aerial":
                self.vae = VAE(input_channels=1, latent_dim=self.latent_space, inference_mode=True)
                WEIGHTS_PATH_AERIAL = os.path.join(BASE_PATH, "weights", self.model_name)
                dict = torch.load(WEIGHTS_PATH_AERIAL, map_location=torch.device(self.device))
                print("[MODEL]: VAE model loaded to device: ", self.device)
                print("[WEIGHTS]: Loading: " + WEIGHTS_PATH_AERIAL)
            elif robot_type == "ground":
                self.vae = VAE_2(input_channels=1, latent_dim=self.latent_space, inference_mode=True)
                WEIGHTS_PATH_GROUND = os.path.join(BASE_PATH, "weights", self.model_name)
                dict = torch.load(WEIGHTS_PATH_GROUND, map_location=torch.device(self.device))
                print("[MODEL]: VAE model loaded to device: ", self.device)
                print("[WEIGHTS]: Loading: " + WEIGHTS_PATH_GROUND)
            else:
                print("[ERROR]: Unknown robot name. I cannot choose vae model")
                return
            self.vae.load_state_dict(dict, strict=True)
            self.vae = self.vae.to(self.device)
        except:
            print("Could not load networks")
            raise Exception("Could not load networks")

    def forward(self, image_numpy):
        self.start_time = time.time()
        self.vae.eval()
        with torch.no_grad():
            torch_image = torch.from_numpy(image_numpy).float().to(self.device)
            # torch_image = torch.clamp(torch_image, 0.0, 1.0)
            x_logit, means, logvar = self.vae(torch_image.view(torch_image.shape[0], torch_image.shape[1], torch_image.shape[2], torch_image.shape[3]))
            reconstructed_image = x_logit
        self.end_time = time.time()
        
        return reconstructed_image, means, logvar, 1000*(self.end_time - self.start_time)
    
    def latent_space_decoded(self, latent_space_vector):
        with torch.no_grad():
            reconstructed_image = self.vae.decode(latent_space_vector)
            return reconstructed_image.cpu().numpy().squeeze(0).squeeze(0)
    
    def get_compute_time(self):
        return (self.end_time - self.start_time)*1000
