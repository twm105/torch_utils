"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""

import torch

# from torch import nn
import torchvision

import matplotlib.pyplot as plt
import numpy as np

import os
import zipfile

from pathlib import Path

from typing import List, Tuple, Dict, Optional

import requests


# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
    Plots linear training data and test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def unnormalise_image(img: torch.Tensor, mean: List, std: List):
    """Reverses the normalisation applied to a tensor image.

    Args:
        img (torch.Tensor): The normalised image tensor of shape (C, H, W).
        mean (List): The mean values used for normalisation, one per channel.
        std (List): The standard deviation values used for normalisation, one per channel.

    Returns:
        torch.Tensor: The unnormalised image tensor.
    """
    mean = torch.tensor(mean).reshape(-1, 1, 1)
    std = torch.tensor(std).reshape(-1, 1, 1)
    return img * std + mean


# show images and labels from dataloader batch
def show_batch_images(
    batch: Tuple[torch.Tensor, torch.Tensor],
    n_images: int = 8,
    images_per_row: int = 8,
    unnormalise: Optional[Dict] = None,
) -> None:
    """Displays a batch of images and their corresponding labels in a grid layout.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): A batch of images and labels from a DataLoader.
        n_images (int, optional): Number of images to display. Defaults to 8.
        images_per_row (int, optional): Number of images per row in the display grid. Defaults to 8.
        unnormalise (Optional[Dict], optional): Dictionary containing 'mean' and 'std' used for unnormalising images. Defaults to None.

    Returns:
        None
    """

    # initiate figure setup
    n_rows = n_images // images_per_row
    fig, axs = plt.subplots(
        n_rows, images_per_row, figsize=(16, 2 * n_rows), sharex=False
    )
    axs = axs.flatten()

    # iterate through subplots, plotting images, adding labels at titles
    X, y = batch
    for i in range(n_images):
        # extract image and label from loader
        image = X[i]
        label = y[i]

        # if cutmix/mixup being used, logits are given for labels, so we need to adapt the title
        if label.ndim == 0:
            title = f"class: {label}"
        elif label.ndim == 1:
            title = ""
            for j, prob in enumerate(label):
                if prob > 0:
                    if len(title) > 0:
                        title += "\n"
                    title += f"{j}: {prob:.3f}"

        # unnormalise if needed
        if unnormalise:
            image = unnormalise_image(
                img=image, mean=unnormalise["mean"], std=unnormalise["std"]
            )

        # construct image
        axs[i].imshow(image.permute(1, 2, 0).clip(0, 1))
        axs[i].set_title(title.strip())
        axs[i].axis("off")

    # tight layout and show
    plt.tight_layout()
    plt.show()


# Plot loss curves of a model
def plot_loss_curves(results, title=None):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # extract values to plot
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    plot_count = 2

    lr_plot = "lr" in results.keys()
    weight_decay_plot = "weight_decay" in results.keys()
    if lr_plot:
        lr = results["lr"]
        if weight_decay_plot:
            weight_decay = results["weight_decay"]
        plot_count += 1

    task_plot = "task" in results.keys()
    if task_plot:
        tasks = results["task"]
        task_map = {
            task: i for i, task in enumerate(list(set(tasks)))
        }  # generate indices for each unique task in results
        task_indices = [
            task_map[task] for task in tasks
        ]  # map indices to task-strings in new list
        plot_count += 1

    if "epochs" in results.keys():
        epochs = results["epochs"]
    else:
        epochs = range(len(results["train_loss"]))

    # setup subplots layout
    n_rows = 1
    n_cols = 2
    if plot_count == 3:
        n_cols = 3
    elif plot_count == 4:
        n_rows = 2

    # configure plot figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 7), sharex=False)
    axs = axs.flatten()

    # Plot loss
    axs[0].plot(epochs, train_loss, label=f"Train Loss ({train_loss[-1]:.4f})")
    axs[0].plot(epochs, test_loss, label=f"Test Loss ({test_loss[-1]:.4f})")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].legend()

    # Plot accuracy
    axs[1].plot(
        epochs, train_accuracy, label=f"Train Accuracy ({train_accuracy[-1]:.4f})"
    )
    axs[1].plot(epochs, test_accuracy, label=f"Test Accuracy ({test_accuracy[-1]:.4f})")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].legend()

    # Plot training config if required
    if lr_plot:
        axs[2].plot(
            epochs, lr, label=f"Learning Rate ({lr[-1]:.3e})"
        )  # always the 3rd plot if present
        if weight_decay_plot:
            axs[2].plot(
                epochs, weight_decay, label=f"Weight Decay ({weight_decay[-1]:.3e})"
            )
        axs[2].set_yscale("log")
        axs[2].set_title("Learning Schedule")
        axs[2].set_xlabel("Epochs")
        axs[2].legend()

    # Plot training tasks if required
    if task_plot:
        i = plot_count - 1  # always the final plot if present
        for task in task_map.keys():
            task_indices_i = [i if i == task_map[task] else None for i in task_indices]
            axs[i].scatter(epochs, task_indices_i, label=task)
        axs[i].set_yticks(ticks=list(task_map.values()), labels=list(task_map.keys()))
        axs[i].set_title("Learning Schedule")
        axs[i].set_xlabel("Epochs")
        axs[i].legend()

    # Finalise plot and show
    plt.tight_layout()
    if title:
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.88)
    plt.show()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".

    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.

    Returns:
        pathlib.Path to downloaded data.

    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)

    return image_path
