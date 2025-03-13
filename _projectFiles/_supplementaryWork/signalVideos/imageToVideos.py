from collections import defaultdict
import cv2
import os
import re
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
from natsort import natsorted

# Max allowed pixels
Image.MAX_IMAGE_PIXELS = 100000000


def resize_with_letterbox(img, frame_size):
    frame_width, frame_height = frame_size
    original_height, original_width = img.shape[:2]

    # Get scale factor to maintain aspect ratio
    scale_factor = min(frame_width / original_width, frame_height / original_height)

    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create blank canvas with target size
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Get top-left coordinates to center the image
    x_offset = (frame_width - new_width) // 2
    y_offset = (frame_height - new_height) // 2

    # Place the resized image onto the canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return canvas


def extract_epoch_number(filename):
    match = re.search(r"epochs(\d+)", filename)
    return int(match.group(1)) if match else 0


def images_to_video(images, video_path, frame_rate=15, video_duration=16, frame_size=(3840, 2160)):
    total_frames_required = frame_rate * video_duration
    num_images = len(images)
    duplication_factor = max(1, round(total_frames_required / num_images))

    print(f"Creating video: {video_path}")
    print(f"Total images: {num_images}, duplicating each frame {duplication_factor} times.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, frame_size)
    frame_count = 0

    for img in images:
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        final_frame = resize_with_letterbox(img_bgr, frame_size)

        for _ in range(duplication_factor):
            video.write(final_frame)
            frame_count += 1

    video.release()
    print(f"Video saved: {video_path}, Total frames: {frame_count}")


def images_to_gif(images, gif_path, duration=100):
    print(f"Creating GIF: {gif_path}")
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"GIF saved: {gif_path}")


if __name__ == "__main__":
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    modelFolderDirectory = os.path.join(currentDirectory, "modelInstances")
    datasetNames = ["wesad", "amigos", "case", "empatch", "dapper", "emognition"]
    initialFolderSubstring = "2025-03-10"

    pattern = re.compile(r"^(.*?)[ _-]?epochs(\d+)\.pdf$", re.IGNORECASE)

    for modelFolderName in os.listdir(modelFolderDirectory):
        if not modelFolderName.startswith(initialFolderSubstring): continue
        modelFolder = os.path.join(modelFolderDirectory, modelFolderName)
        if not os.path.isdir(modelFolder): continue

        for datasetFolderName in os.listdir(modelFolder):
            if datasetFolderName.lower() not in datasetNames: continue
            datasetFolder = os.path.join(modelFolder, datasetFolderName)

            signalEncodingFolder = os.path.join(datasetFolder, "signalEncoding")
            signalReconstruction = os.path.join(datasetFolder, "signalReconstruction")

            for pdfFolder in [signalEncodingFolder, signalReconstruction]:
                if not os.path.exists(pdfFolder): continue

                pdf_files = [f for f in os.listdir(pdfFolder) if f.endswith(".pdf")]
                pdf_groups = defaultdict(list)

                for pdf_file in pdf_files:
                    match = pattern.match(pdf_file)
                    if match:
                        prefix, epoch = match.groups()
                        pdf_groups[prefix].append((int(epoch), pdf_file))

                for prefix, files in pdf_groups.items():
                    files = natsorted(files, key=lambda x: x[0])  # Sort by epoch
                    images = []

                    for _, pdf_file in files:
                        pdf_path = os.path.join(pdfFolder, pdf_file)
                        try:
                            img = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)[0]
                            images.append(img)
                        except Exception as e:
                            print(f"Error processing {pdf_file}: {e}")

                    if images:
                        video_path = os.path.join(pdfFolder, f"{prefix}.mp4")
                        images_to_video(images, video_path)

                        # gif_path = os.path.join(pdfFolder, f"{prefix}.gif")
                        # images_to_gif(images, gif_path)
