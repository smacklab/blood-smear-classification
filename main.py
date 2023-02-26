import os
import ultralytics
from ultralytics import YOLO
import numpy
import numpy as np
import tifffile
from PIL import Image, ImageOps
from tqdm import tqdm

model = YOLO("best_model_checkpoint_Feb25.pt")
if not os.path.exists("output"):
    os.makedirs("output")


def create_output_folders(image_name):
    if not os.path.exists("output/" + image_name):
        os.makedirs("output/" + image_name)

    # create folders for each class Dense, Good, Sparse
    for class_name in model.names.values():
        if not os.path.exists("output/" + image_name + "/" + class_name):
            os.makedirs("output/" + image_name + "/" + class_name)


def save_to_folder(results, cropped_image, cropped_image_name):
    parsed_results = list([float(x) for x in results[0].probs])
    max_index = np.argmax(parsed_results)
    class_name = model.names[max_index]
    # save image to folder
    save_path = os.path.join("output", image_name,
                             class_name, cropped_image_name)
    cropped_image.save(save_path, "JPEG", quality=100)


if __name__ == "__main__":
    for filename in os.listdir("data"):
        f = os.path.join("data", filename)
        print(f)
        if not os.path.isfile(f) or not f.endswith(".ndpi"):
            continue
        with tifffile.TiffFile(f) as tif:
            image = tif.asarray()
            image = Image.fromarray(image)
            w, h = image.size
            print("NDPI Image size: {}x{}".format(w, h))
            image_name = os.path.splitext(filename)[0]
            create_output_folders(image_name)

            # split image into 512x512 tiles, column by column
            for c in tqdm(range(0, w, 512)):
                for r in tqdm(range(0, h, 512), leave=False):
                    cropped_image = image.crop((c, r, c + 512, r + 512))
                    cropped_image_name = image_name + \
                        "_cropped_" + str(c) + "_" + str(r) + ".jpg"
                    # grayscaled_image = ImageOps.grayscale(cropped_image)

                    # run prediction on image
                    results = model(cropped_image)
                    save_to_folder(results, cropped_image, cropped_image_name)
