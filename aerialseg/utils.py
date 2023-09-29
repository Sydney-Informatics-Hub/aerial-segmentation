from PIL import Image
import os

def save_images_as_gif(input_folder, output_gif_path, duration=100):
    """
    Save a folder of images as an animated GIF.

    Args:
    - input_folder (str): Path to the folder containing image files (e.g., JPEG or PNG).
    - output_gif_path (str): Path to save the animated GIF file.
    - duration (int, optional): Duration (in milliseconds) for each frame in the GIF. Default is 100ms.

    Returns:
    - None
    """
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]

    if not image_files:
        print("No image files found in the input folder.")
        return

    images = []
    for image_file in sorted(image_files):
        image_path = os.path.join(input_folder, image_file)
        img = Image.open(image_path)
        images.append(img)

    # Save the animated GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
