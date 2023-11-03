import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from PIL import Image
from pathlib import Path
import zipfile


# Plot image sample
def plot_sample(
    sample: torch.Tensor,
    label: str):

    """PLot an image (tensor) using matplotlib.pyplot"""

    plt.imshow(sample.permute(1,2,0))
    plt.title(f"Label: {label}")
    plt.xlabel(f"dimensions: {sample.permute(1,2,0).shape} (hwc)")
    plt.show()

# Calculate mean and std
def cal_mean_std(
    train: Dataset,
    test: Dataset
    ):

    """Calulate mean and std for torchvision.transforms"""

    mean, std = 0., 0.
    mean_t, std_t = 0., 0.

    total_samples, total_samples_t = len(train), len(test)

    for data, _ in train:
        img = data.numpy()
        mean += np.mean(img, axis=(1, 2)) # (0, 1) hwc or (1, 2) chw
        std += np.std(img, axis=(1, 2))

    mean /= total_samples
    std /= total_samples


    for data, _ in test:
        img = data.numpy()
        mean_t += np.mean(img, axis=(1, 2)) # (0, 1) hwc or (1, 2) chw
        std_t += np.std(img, axis=(1, 2))

    mean_t /= total_samples_t
    std_t /= total_samples_t

    return (mean, std), (mean_t, std_t)

# Create dataloader
def make_dataloader(
    train: Dataset,
    test: Dataset,
    num_workers: int,
    batch_size: int = 32):

    """Create dataloader for train and test"""

    train_dataloader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader

# save model
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

# pil images to torch tensor 
def load_custom_samples_from_dir(
    dir: str,
    transform: torchvision.transforms = transforms.ToTensor()) -> list:

  """Load images as torch tensors from a local directory"""

  # store images
  imgs = []

  # Iter through dir
  for i in os.listdir(dir):
    # create path
    img_dir = os.path.join(dir, i)

    img = Image.open(img_dir)
    tensor_img = transform(img)
    imgs.append(tensor_img)

  return imgs

# download zip
def download_data(
    source: str,
    target_dir: Path,
    remove_zip: bool = True,
    zip_name: str = "content"
    ):
  """
  Download a zip file and extract it
  Note: if target_dir exist, download does not work
  """
  # create target directory
  if target_dir.is_dir():
    print(f"[INFO] Folder {target_dir} already exists, skipping process")
  else:
    print(f"[INFO] Folder {target_dir} doesn't exists, creating one..")
    target_dir.mkdir(parents=True, exist_ok=True)

    # download content
    with open(target_dir / f"{zip_name}.zip", "wb") as f:
      request = requests.get(source)
      print(f"Download {zip_name} packege...")
      f.write(request.content)
      print("[INFO] File downloaded")

      print(target_dir / f"{zip_name}.zip")
    # extract content
    with zipfile.ZipFile(target_dir / f"{zip_name}.zip", "r") as zip_ref:
      print(f"Unzipping content...")
      zip_ref.extractall((target_dir/zip_name))
      print(f"[INFO] {zip_name} unzipped")

    # Remove zip
    if remove_zip:
      print("removing zip file...")
      os.remove(target_dir / f"{zip_name}.zip")
      print("[INFO] File zip revomed successfully")

# make predictions function
def plot_predictions(
    data: list,
    model: torch.nn.Module,
    device: str,
    num_samples: int = 5,):

  """PLots the image from a package of data,
  model predict them and plot its answer
  [Args]
  data: list of pyorch tensors, model: nn.module model,
  num_samples: how many images it will plot (max 10 samples)"""

  if num_samples > 10:
    print(f"[INFO] there are too many samples ({num_samples}), change to 10 (max)")
    num_samples = 10

  for index, sample in enumerate(data):
    if not index + 1 > 10:
      if not index + 1 > num_samples:
        # predict
        print(sample.unsqueeze(dim=0).shape)
        logit = model(sample.unsqueeze(dim=0).to(device))
        pred = torch.argmax(torch.softmax(logit.squeeze(), dim=0))
        # plot
        plt.imshow(sample.permute(1,2,0))
        plt.title(f"predicted label: {pred}")
        plt.show()
      else:
        break
    else:
      break
