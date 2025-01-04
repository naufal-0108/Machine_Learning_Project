import cv2, os
import numpy as np
import torch

def find_closest_coords(input_tuple, tuple_list):
    input_sum = sum(input_tuple)
    # Find the closest tuple by calculating the sum differences and picking the minimum
    closest_tuple = min(tuple_list, key=lambda t: abs(sum(t) - input_sum))
    return closest_tuple

def get_text_dimensions(text, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=1):
    """
    Get the width and height of a text string in pixels.
    
    Parameters:
    text (str): The text string to measure
    font_face: OpenCV font type (default: cv2.FONT_HERSHEY_SIMPLEX)
    font_scale (float): Font scale factor (default: 1)
    thickness (int): Thickness of the lines used to draw the text (default: 1)
    
    Returns:
    tuple: (width, height) in pixels
    """
    # Get the text size
    (width, height), baseline = cv2.getTextSize(
        text, 
        font_face,
        font_scale,
        thickness
    )
    
    return width, height

def create_transparent_polygon(image, polygons, color=(0, 255, 0), alpha=0.5):
    """
    Create a transparent polygon on an image without affecting image opacity
    Args:
        image: Base image
        points: List of polygon points [(x1,y1), (x2,y2), ...]
        color: Polygon color in BGR
        alpha: Transparency value (0: transparent, 1: opaque)
    Returns:
        Image with transparent polygon
    """
    # Create mask for the polygon
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    colored_polygon = np.zeros_like(image)

    for polygon_points in polygons:
        cv2.fillPoly(mask, [polygon_points], 255)
        cv2.fillPoly(colored_polygon, [polygon_points], (0,0,0))

    # Blend only the polygon area
    result = image.copy()
    polygon_area = mask > 0
    result[polygon_area] = cv2.addWeighted(
        image[polygon_area],
        1 - alpha,
        colored_polygon[polygon_area],
        alpha,
        0
    )

    # Draw polygon edges

    for polygon_points in polygons:
        cv2.polylines(result, [polygon_points], True, color, 3)
    
    return result


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load model checkpoint.

    Args:
        model: PyTorch model to be loaded.
        optimizer: Optimizer used for training.
        checkpoint_path: Path to the checkpoint file.
    """

    if scheduler == None:
      if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        val_dice = checkpoint['val_dice']
        print(f"Checkpoint loaded. Epoch: {epoch}, Validation Loss: {val_loss}")
        return model, optimizer, epoch, val_loss, val_dice
      else:
          raise FileNotFoundError("Checkpoint file not found.")


    else:

      if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        print(f"Checkpoint loaded. Epoch: {epoch}, Validation Loss: {val_loss}")
        return model, optimizer, scheduler, epoch, val_loss

      else:
        raise FileNotFoundError("Checkpoint file not found.")

    
      