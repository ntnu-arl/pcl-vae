import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Warp
import warp as wp
wp.init()

# Occupancy map
VOXEL_FREE = 0
VOXEL_OCCUPIED = 1
VOXEL_UNKNOWN = 2

# Range image parameters
INVALID_PIXEL_VALUE = -1.0


def occupancy_loss(x, x_logit, mean, logvar, latent_dims, voxel_size, max_range, min_range, h_fov, v_fov, alpha = 1.0, beta_coeff= 1.0, gamma=1.0):

    # Scale up the normalized range image
    original_range_image = torch.where(x > 0, x * max_range, x)
   
    # Re-projected range image (range image -> point cloud -> occupancy map -> range image)
    reprojected_range_image = range_image_reprojection(original_range_image, voxel_size, max_range, h_fov, v_fov)
    voxelized_range_image = reprojected_range_image.clone()

    # Normalize the reprojected range image
    reprojected_range_image_normalized = reprojected_range_image.clone()
    reprojected_range_image_normalized[reprojected_range_image_normalized > max_range] = max_range        # max_range | INVALID_PIXEL_VALUE
    reprojected_range_image_normalized[reprojected_range_image_normalized < min_range] = INVALID_PIXEL_VALUE
    reprojected_range_image_normalized = reprojected_range_image_normalized / max_range
    reprojected_range_image_normalized[reprojected_range_image_normalized < 0] = -2*INVALID_PIXEL_VALUE     # INVALID_PIXEL_VALUE | -2*INVALID_PIXEL_VALUE

    # display_range_images_for_comparison(original_range_image, reprojected_range_image)
    
    # Reconstruction loss
    valid_pixel_mask = torch.where((reprojected_range_image_normalized > 0) & (reprojected_range_image_normalized <= 1), torch.ones_like(x), torch.zeros_like(x))
    REC_MSE_LOSS = nn.MSELoss(reduction="none")
    cross_ent_rec = REC_MSE_LOSS(x_logit, reprojected_range_image_normalized)* valid_pixel_mask
    reconstruction_loss = torch.mean(torch.sum(cross_ent_rec, dim=[1, 2, 3]))

    # KL-Divergence loss
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    beta_norm = (beta_coeff*latent_dims)/(x.shape[2]*x.shape[3]) 
    kld_loss = kld_loss * beta_norm

    loss = alpha * reconstruction_loss + kld_loss
    return loss, reconstruction_loss, kld_loss, voxelized_range_image


def range_image_reprojection(range_img, voxel_size, max_range, h_fov, v_fov):
    batch_size = range_img.shape[0]
    channels = range_img.shape[1]
    img_height = range_img.shape[2]
    img_width = range_img.shape[3]
    
    # Initialize a list to store the reprojected range images for each batch
    reprojected_range_images = []
    
    for i in range(batch_size):
        # Select the range image for the current batch item (assume single-channel input)
        single_range_img = range_img[i].unsqueeze(0)  # Shape: (1, 1, H, W)

        # Convert range image to point cloud (N,3)
        point_cloud = convert_range_image_to_pointcloud(single_range_img, h_fov, v_fov)

        # Convert point cloud to occupancy map (e.g., 101, 101, 101)
        occupancy_map = convert_point_cloud_to_occupancy_map(point_cloud, voxel_size, max_range)

        # Reproject the occupancy map to a range image
        reprojected_range_image = convert_occupancy_map_to_range_image(
            point_cloud, occupancy_map, h_fov, v_fov, img_width, img_height, voxel_size, max_range
        )

        # Ensure the output is (1, H, W) shape, remove any extra dimensions
        if reprojected_range_image.dim() == 4:
            reprojected_range_image = reprojected_range_image.squeeze(1)  # Squeeze if extra channel dim is present

        # Add the reprojected image to the list
        reprojected_range_images.append(reprojected_range_image)

    # Stack all the reprojected range images along the batch dimension
    reprojected_range_images = torch.stack(reprojected_range_images, dim=0)

    return reprojected_range_images


def convert_range_image_to_pointcloud(range_image, horizontal_fov, vertical_fov):
    """
    Convert a LiDAR range image into a 3D point cloud using Warp kernel.

    Args:
        range_image (torch.Tensor): The 4D range image tensor with shape (1, 1, H, W).
        horizontal_fov (float): The horizontal field of view of the LiDAR in radians.
        vertical_fov (float): The vertical field of view of the LiDAR in radians.

    Returns:
        torch.Tensor: Nx3 tensor where N is the number of valid points.
    """

    height, width = range_image.shape[2], range_image.shape[3]

    # Convert the range_image to a Warp array with 2 dimensions
    range_image_wp = wp.from_torch(range_image.squeeze(dim=0).squeeze(dim=0), dtype=wp.float32)
    
    # Initialize the output Warp array for the point cloud
    points_wp = wp.zeros(height * width, dtype=wp.vec3)

    # Launch the Warp kernel
    wp.launch(
        kernel=convert_range_image_to_pointcloud_kernel,
        dim=height * width,
        inputs=[range_image_wp, points_wp, width, height, horizontal_fov, vertical_fov]
    )

    # Convert the Warp array to a Pytorch tensor
    points = wp.to_torch(points_wp)
    
    return points


@wp.kernel
def convert_range_image_to_pointcloud_kernel(
    range_image: wp.array2d(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    horizontal_fov: float,
    vertical_fov: float):
    
    u = wp.tid() % width    # id of columns
    v = wp.tid() // width   # id of rows

    r = range_image[v, u]

    if r > 0.0:  # Valid ranges
        # Ensure calculations are done with consistent float32 types
        u_f = wp.float32(u)
        v_f = wp.float32(v)
        width_f = wp.float32(width - 1)
        height_f = wp.float32(height - 1)
        
        # Calculate the angles
        horizontal_angle = (u_f / width_f) * horizontal_fov - (horizontal_fov / 2.0)
        vertical_angle = (v_f / height_f) * vertical_fov - (vertical_fov / 2.0)
        
        # Convert polar coordinates (range, horizontal_angle, vertical_angle) to Cartesian coordinates (x, y, z)
        x = r * wp.cos(vertical_angle) * wp.cos(horizontal_angle)
        y = -r * wp.cos(vertical_angle) * wp.sin(horizontal_angle)
        z = -r * wp.sin(vertical_angle)
    elif r == INVALID_PIXEL_VALUE:
        x = wp.nan
        y = wp.nan
        z = wp.nan
    
    # Set the point in the output array
    points[v * width + u] = wp.vec3(x, y, z)


def convert_point_cloud_to_occupancy_map(point_cloud, voxel_size, max_range):
    """
    Convert a point cloud to an occupancy map using Warp kernel.

    Args:
        point_cloud (torch.Tensor): The point cloud as an Nx3 tensor.
        voxel_size (float): The size of each voxel in the grid.

    Returns:
        torch.Tensor: The 3D occupancy map as a torch tensor.
    """

    # Grid size
    grid_size = wp.vec3i(
        int(max_range)*int(max_range) + 1,
        int(max_range)*int(max_range) + 1,
        int(max_range)*int(max_range) + 1
    )

    # Grid center
    grid_center = wp.vec3f(
        voxel_size * grid_size.x / 2.0,
        voxel_size * grid_size.y / 2.0,
        voxel_size * grid_size.z / 2.0
    )

    # Convert point cloud to Warp array (129600,)
    points_wp = wp.from_torch(point_cloud, dtype=wp.vec3)

    # Initialize the occupancy map with VOXEL_UNKNOWN
    occupancy_map_wp = wp.full(grid_size, VOXEL_UNKNOWN, dtype=wp.int32)

    # Launch the kernel to create occupancy map
    wp.launch(
        kernel=convert_point_cloud_to_occupancy_map_free_kernel, 
        dim=len(points_wp), 
        inputs=[points_wp, occupancy_map_wp, voxel_size, grid_size, grid_center])
    
    wp.launch(
        kernel=convert_point_cloud_to_occupancy_map_occupied_kernel, 
        dim=len(points_wp), 
        inputs=[points_wp, occupancy_map_wp, voxel_size, grid_size, grid_center])
  
    # Convert the Warp array to a Pytorch tensor
    occupancy_map = wp.to_torch(occupancy_map_wp)

    return occupancy_map


@wp.kernel
def convert_point_cloud_to_occupancy_map_free_kernel(
    points: wp.array(dtype=wp.vec3), 
    occupancy_map: wp.array3d(dtype=wp.int32), 
    voxel_size: float, 
    grid_size: wp.vec3i,
    grid_center: wp.vec3f):
    
    tid = wp.tid()  # Thread ID in the grid

    # Load the point from the adjusted point cloud
    point = points[tid]
    
    # Check if the point contains NaN values
    if wp.isnan(point.x) or wp.isnan(point.y) or wp.isnan(point.z):
        return  # Do nothing if the point is NaN
    
    # Normalize direction vector
    direction = wp.normalize(point)
    
    # Ray march through the voxel grid
    max_distance = wp.length(point)
    t = float(0.0)  # Declare t as a mutable float

    while t < max_distance:
        # Calculate the current position along the ray
        pos = direction * t
        
        # Convert the position to voxel grid coordinates
        voxel_coord_x = wp.int32((pos.x + grid_center.x) / voxel_size)
        voxel_coord_y = wp.int32((pos.y + grid_center.y) / voxel_size)
        voxel_coord_z = wp.int32((pos.z + grid_center.z) / voxel_size)

        # Ensure the coordinates are within the grid bounds
        if (0 <= voxel_coord_x < grid_size.x and
            0 <= voxel_coord_y < grid_size.y and
            0 <= voxel_coord_z < grid_size.z):

            # Mark the voxel as occupied
            occupancy_map[voxel_coord_x, voxel_coord_y, voxel_coord_z] = VOXEL_FREE
            
        # Increment t 
        t = t + voxel_size / 2.0  


@wp.kernel
def convert_point_cloud_to_occupancy_map_occupied_kernel(
    points: wp.array(dtype=wp.vec3), 
    occupancy_map: wp.array3d(dtype=wp.int32), 
    voxel_size: float, 
    grid_size: wp.vec3i,
    grid_center: wp.vec3f):
    
    tid = wp.tid()  # Thread ID in the grid

    # Load the point from the adjusted point cloud
    point = points[tid]
    
    # Check if the point contains NaN values
    if wp.isnan(point.x) or wp.isnan(point.y) or wp.isnan(point.z):
        return  # Do nothing if the point is NaN
    
    # Mark the end-point voxel as occupied
    voxel_coord_x = wp.int32((point.x + grid_center.x) / voxel_size)
    voxel_coord_y = wp.int32((point.y + grid_center.y) / voxel_size)
    voxel_coord_z = wp.int32((point.z + grid_center.z) / voxel_size)

    # Ensure the coordinates are within the grid bounds
    if (0 <= voxel_coord_x < grid_size.x and
        0 <= voxel_coord_y < grid_size.y and
        0 <= voxel_coord_z < grid_size.z):

        # Mark the voxel as occupied
        occupancy_map[voxel_coord_x, voxel_coord_y, voxel_coord_z] = VOXEL_OCCUPIED


def convert_occupancy_map_to_range_image(point_cloud, occupancy_map, horizontal_fov, vertical_fov, img_width, img_height, voxel_size, max_range):
    """
    Convert an occupancy map back to a range image using Warp kernel.

    Args:
        occupancy_map (torch.Tensor): The 3D occupancy map tensor with shape (101, 101, 101).
        horizontal_fov (float): The horizontal field of view of the sensor in radians.
        vertical_fov (float): The vertical field of view of the sensor in radians.

    Returns:
        torch.Tensor: The 2D range image tensor.
    """
    
    # Convert point cloud to a Warp array with the correct type
    points_wp = wp.from_torch(point_cloud, dtype=wp.vec3)

    # Convert the occupancy map to a Warp array with the correct type
    occupancy_map_wp = wp.from_torch(occupancy_map, dtype=wp.int32)
    grid_size = wp.vec3i(occupancy_map.shape[0], occupancy_map.shape[1], occupancy_map.shape[2])

    # Initialize the reprojected range image
    range_image_wp = wp.full((img_height, img_width), max_range, dtype=wp.float32)

    # Launch the kernel
    wp.launch(
        kernel=occupancy_map_to_range_image_kernel,
        dim=img_height *img_width,
        inputs=[range_image_wp, points_wp, occupancy_map_wp, max_range, voxel_size, grid_size, horizontal_fov, vertical_fov, img_width, img_height]
    )
    
    # Convert the Warp array to a Pytorch tensor
    range_image = wp.to_torch(range_image_wp)
    range_image = range_image.unsqueeze(0).unsqueeze(0)

    return range_image


@wp.kernel
def occupancy_map_to_range_image_kernel(
    range_image: wp.array2d(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3), 
    occupancy_map: wp.array3d(dtype=wp.int32),
    max_range: float,
    voxel_size: float,
    grid_size: wp.vec3i,
    horizontal_fov: float,
    vertical_fov: float,
    width: int,
    height: int):

    u = wp.tid() % width  # Horizontal index in the range image
    v = wp.tid() // width  # Vertical index in the range image

    pt = points[v * width + u]

    # Check if the point contains NaN values
    if wp.isnan(pt.x) or wp.isnan(pt.y) or wp.isnan(pt.z):
        range_image[v,u] = wp.float32(INVALID_PIXEL_VALUE)
        return  # Do nothing if the point is NaN
    else:
        sensor_position = wp.vec3(
            voxel_size * float(grid_size.x) / 2.0,
            voxel_size * float(grid_size.y) / 2.0,
            voxel_size * float(grid_size.z) / 2.0
        )

        voxel_coord_x = wp.int32((pt.x + sensor_position.x) / voxel_size)
        voxel_coord_y = wp.int32((pt.y + sensor_position.y) / voxel_size)
        voxel_coord_z = wp.int32((pt.z + sensor_position.z) / voxel_size)
        
        if (0 <= voxel_coord_x < grid_size.x and 
            0 <= voxel_coord_y < grid_size.y and 
            0 <= voxel_coord_z < grid_size.z): 
        
            voxel_center = wp.vec3(
                (float(voxel_coord_x) + 0.5) * voxel_size,
                (float(voxel_coord_y) + 0.5) * voxel_size,
                (float(voxel_coord_z) + 0.5) * voxel_size
            )

            # Calculate the difference vector between point_a and point_b
            difference = voxel_center - sensor_position

            # Calculate the squared distance
            squared_distance = difference.x * difference.x + difference.y * difference.y + difference.z * difference.z

            # Compute the Euclidean distance by taking the square root of the squared distance
            r = wp.sqrt(squared_distance)
            # r = wp.length(pt)
            range_image[v,u] = wp.float32(r)


def display_range_images_for_comparison(img_1, img_2):
    
    numpy_array_1 = img_1.detach().cpu().numpy()    # Convert to NumPy, detach and move to CPU if necessary
    numpy_array_1 = numpy_array_1[0, 0, :, :]       # Shape will be [64, 512]

    numpy_array_2 = img_2.detach().cpu().numpy()    # Convert to NumPy, detach and move to CPU if necessary
    numpy_array_2 = numpy_array_2[0, 0, :, :]  # Shape will be [64, 512]
  
    # Normalize the array to 0-255 if necessary
    numpy_array_1 = cv2.normalize(numpy_array_1, None, 0, 255, cv2.NORM_MINMAX)
    numpy_array_2 = cv2.normalize(numpy_array_2, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to unsigned 8-bit integer
    numpy_array_1 = numpy_array_1.astype(np.uint8)
    numpy_array_2 = numpy_array_2.astype(np.uint8)

    # Display the image using OpenCV
    cv2.imshow('Image 1', numpy_array_1)
    cv2.imshow('Image 2', numpy_array_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()