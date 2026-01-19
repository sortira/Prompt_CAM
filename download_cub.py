import os
import shutil
import tarfile
import urllib.request

# Configuration
URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
DEST_DIR = "datasets"
CUB_DIR_NAME = "cub"
TGZ_FILE = "CUB_200_2011.tgz"


def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = (downloaded / total_size) * 100
    print(f"\rDownloading: {percent:.2f}%", end="")


def setup_cub():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    tgz_path = os.path.join(DEST_DIR, TGZ_FILE)

    # 1. Download
    if not os.path.exists(tgz_path):
        print(f"Downloading CUB dataset to {tgz_path}...")
        try:
            urllib.request.urlretrieve(URL, tgz_path, report_progress)
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nError downloading: {e}")
            return
    else:
        print("Dataset archive already exists. Skipping download.")

    # 2. Extract
    print("Extracting archive (this may take a while)...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=DEST_DIR)

    # The extracted folder is named 'CUB_200_2011'
    extracted_root = os.path.join(DEST_DIR, "CUB_200_2011")
    target_root = os.path.join(DEST_DIR, CUB_DIR_NAME)

    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    os.makedirs(target_root)

    images_txt_path = os.path.join(extracted_root, "images.txt")
    split_txt_path = os.path.join(extracted_root, "train_test_split.txt")

    # 3. Read Metadata
    image_paths = {}
    with open(images_txt_path, "r") as f:
        for line in f:
            id_, path = line.strip().split()
            image_paths[id_] = path

    is_train = {}
    with open(split_txt_path, "r") as f:
        for line in f:
            id_, val = line.strip().split()
            # 1 = train, 0 = test (in CUB documentation)
            # However, standard practice usually treats 'test' as 'val' for simple splits
            is_train[id_] = val == "1"

    # 4. Reorganize
    print("Reorganizing files into train/val structure...")

    train_dir = os.path.join(target_root, "train")
    val_dir = os.path.join(target_root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    count = 0
    total = len(image_paths)

    for img_id, rel_path in image_paths.items():
        src_path = os.path.join(extracted_root, "images", rel_path)

        # Determine destination
        # rel_path looks like: 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
        # We want to keep the class folder structure

        class_name, filename = os.path.split(rel_path)

        if is_train[img_id]:
            dest_folder = os.path.join(train_dir, class_name)
        else:
            dest_folder = os.path.join(val_dir, class_name)

        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy2(src_path, os.path.join(dest_folder, filename))

        count += 1
        if count % 100 == 0:
            print(f"\rProcessed {count}/{total} images", end="")

    print(f"\n\nDataset successfully setup at: {target_root}")
    print("You can now clean up the tar file and extracted folder if you wish.")
    print(f"Path to use in --data_path: {target_root}")


if __name__ == "__main__":
    setup_cub()
