import cv2
import os
import re
from pdf2image import convert_from_path
import numpy as np

from PIL import Image

# max allowed pixels
Image.MAX_IMAGE_PIXELS = 100000000


def resize_with_letterbox(img, frame_size):
    # letter box padding to keep the aspect ratio
    frame_width, frame_height = frame_size
    original_height, original_width = img.shape[:2]

    # get the scale factor to keep the image in frame
    scale_factor = min(frame_width / original_width, frame_height / original_height)

    # get the new dimension
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # image resize
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # blank canvas generation with targeted frame sizes
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # canvas[:] = (255, 255, 255) # to change canvas color if needed

    # get the top-left coordinates to center the resized image
    x_offset = (frame_width - new_width) // 2
    y_offset = (frame_height - new_height) // 2

    # place the resized image onto the canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return canvas


def extract_epoch_number(filename):
    # sort by epochs
    match = re.search(r"epochs(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def images_to_video(pdf_files, pdf_folder, video_name, video_duration=5, frame_rate=60, frame_size=(3840, 2160)):
    # Converts PDF images to a video.

    # if the total number of PDF images is less than the required frame count (frame_rate * video_duration),
    # each frame is duplicated to reach the desired video duration.

    total_frames_required = frame_rate * video_duration  # duplicate frames
    num_images = len(pdf_files)
    duplication_factor = total_frames_required / num_images
    duplication_factor_int = int(round(duplication_factor))
    print(f"Total images: {num_images}, duplicating each frame {duplication_factor_int} times to reach around{total_frames_required} frames.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # four character code for encoding video files; hmp4 format, others format also supported check openCV documentation
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)

    frame_count = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        print(f"{pdf_file}: {len(images)} page(s) converted.")

        for img in images:
            # convert PIL (python imaging library: used for opening and manipulating images) image to numpy array and change color space from RGB to BGR for OpenCV
            img_np = np.array(img)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_np, frame_size)

            # Write the same frame multiple times with duplication factor
            for _ in range(duplication_factor_int):
                video.write(img_resized)
                frame_count += 1

    video.release()
    print(f"Video saved at: {video_name}")
    print(f"Total frames written: {frame_count}")


def images_to_video_custom(pdf_files, pdf_folder, video_name, video_duration=5, frame_rate=60, frame_size=(3840, 2160)):
    total_frames_required = frame_rate * video_duration
    num_images = len(pdf_files)
    duplication_factor = total_frames_required / num_images
    duplication_factor_int = int(round(duplication_factor))
    print(f"Total images: {num_images}, duplicating each frame {duplication_factor_int} times to reach ~{total_frames_required} frames.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)
    frame_count = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        print(f"{pdf_file}: {len(images)} page(s) converted.")

        for img in images:
            img_np = np.array(img)
            # Convert from RGB (PIL) to BGR (OpenCV)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            # Instead of directly resizing, use the letterbox function
            final_frame = resize_with_letterbox(img_bgr, frame_size)
            for _ in range(duplication_factor_int):
                video.write(final_frame)
                frame_count += 1

    video.release()
    print(f"Video saved at: {video_name}")
    print(f"Total frames written: {frame_count}")


if __name__ == "__main__":
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    modelFolderDirectory = os.path.join(currentDirectory, "modelInstances")
    datasetNames = ["wesad", "amigos", "case", "empatch", "dapper", "emognition"]
    initialFolderSubstring = "2025-03-10"

    # For each model in the folder.
    for modelFolderName in os.listdir(modelFolderDirectory):
        if not modelFolderName.startswith(initialFolderSubstring): continue
        modelFolder = os.path.join(modelFolderDirectory, modelFolderName)
        
        # For each dataset in the folder.
        for datasetFolderName in os.listdir(modelFolder):
            if datasetFolderName.lower() not in datasetNames: continue
            datasetFolder = os.path.join(modelFolder, datasetFolderName)

            # Get the two subfolders with all the files.
            signalEncodingFolder = os.path.join(datasetFolder, "signalEncoding")
            signalReconstruction = os.path.join(datasetFolder, "signalReconstruction")

            # Get all the
            pdf_files = sorted([file for file in os.listdir(pdf_folder) if file.endswith(".pdf")], key=extract_epoch_number)

    pdf_files = get_pdf_files(pdf_folder)

    video_duration = 5
    frame_rate = 60
    frame_size = (3840, 2160)

    images_to_video_custom(pdf_files, pdf_folder, video_name, video_duration, frame_rate, frame_size)
