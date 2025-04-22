"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch.utils.tensorboard import SummaryWriter

from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone
from itertools import product
from tqdm.auto import tqdm
import functools
import os

from utils import save_model, save_checkpoint


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler = None,
    use_bf16: bool = True,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        scheduler: An optional PyTorch lr_scheduler.
        use_bf16: Boolean that turns on using BF16 if True (default True).

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # configure autocast dtype to utilise bf16 for forward pass if available (A100 or more recent)
    model_dtype = next(model.parameters()).dtype
    fallback_dtype = (
        model_dtype
        if model_dtype in [torch.float16, torch.bfloat16, torch.float32]
        else torch.float32
    )
    autocast_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported() and use_bf16
        else fallback_dtype
    )

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # set to bf16 if GPU supports it (just fwd pass and loss calc, backward inherits from fwd pass)
        with torch.autocast(
            device_type=device,
            dtype=autocast_dtype,
        ):
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # LR scheduler step
    scheduler.step()

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_bf16: bool = True,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        use_bf16: Boolean that turns on using BF16 if True (default True).

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # configure autocast dtype to utilise bf16 for forward pass if available (A100 or more recent)
    model_dtype = next(model.parameters()).dtype
    fallback_dtype = (
        model_dtype
        if model_dtype in [torch.float16, torch.bfloat16, torch.float32]
        else torch.float32
    )
    autocast_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported() and use_bf16
        else fallback_dtype
    )

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # set to bf16 if GPU supports it (just fwd pass and loss calc, backward inherits from fwd pass)
            with torch.autocast(
                device_type=device,
                dtype=autocast_dtype,
            ):
                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def create_writer(
    experiment_name: str = None, model_name: str = None, extra: str = None
) -> torch.utils.tensorboard.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current datetime in YYMMDDzHHMM format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now(timezone.utc).strftime(
        r"%y%m%dz%H%M"
    )  # returns current datetime in YYMMDDzHHMM format

    # initialise the path structure for log_dir
    log_dir_path = ["runs", timestamp]

    # add optional path structure as required by function input params
    if experiment_name:
        log_dir_path.append(experiment_name)
    if model_name:
        log_dir_path.append(model_name)
    if extra:
        log_dir_path.append(extra)

    # set log_dir based on config above
    log_dir = os.path.join(*log_dir_path)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    writer: torch.utils.tensorboard.writer.SummaryWriter = None,
    torch_compile: bool = True,
    use_bf16: bool = True,
    float32_matmul_precision: str = "high",
    checkpoint_interval: int = None,
    save_final_model: bool = False,
    model_save_path: str = None,
    model_save_base_name: str = None,
    colab_local_path: str = None,
    config_file: str = None,
    results: Dict = None,
    task: str = None,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch model through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Also supports checkpointing and saving of final model state, with
    customisable save paths and naming.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimise the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        scheduler: An optional LR scheduler.
        scaler: An optional GradScaler for mixed precision training.
        writer: A TensorBoard SummaryWriter instance for tracking experiments.
        float32_matmul_precision: Sets matmul precision (default "High", TF32)
        torch_compile: Applies torch.compile() to the model if True (default True)
        use_bf16: Boolean that turns on using BF16 if True (default True).
        checkpoint_interval: If not None, model checkpoints are saved every `checkpoint_interval` epochs.
        save_final_model: If True, final model weights will be saved after all epochs complete.
        model_save_path: Path for saving model checkpoints and final weights.
        model_save_base_name: Stem for naming saved models (e.g. base_name_cp01.pth).
        colab_local_path: Optional path to sync checkpoints on SSD prior to copying to Google Colab Drive.
        config_file: Optional config file to store with checkpoint.
        results: An optional results dictionary to extend training results.
        task: A string identifying the training objective or dataset (e.g. "CIFAR10").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form:
            {
                train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...],
                lr: [...],
                weight_decay: [...],
                task: [...],
                epoch: [...]
            }
        For example if training for epochs=2:
            {
                train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973],
                lr: [...],
                weight_decay: [...],
                task: [...],
                epoch: [1, 2]
            }
    """
    # Input checks
    if checkpoint_interval or save_final_model:
        # check that save path is defined
        if model_save_path is None:
            print(
                "[WARNING] Model checkpoint interval provided with no save_path. Checkpoints will not be saved."
            )
        if model_save_base_name is None:
            # Get timestamp of current date (all experiments on certain day live in same folder)
            timestamp = datetime.now(timezone.utc).strftime(
                r"%y%m%dz%H%M"
            )  # returns current datetime in YYMMDDzHHMM format
            model_save_base_name = timestamp + "-model"
            print(
                f"[WARNING] No model_save_base_name provided, defaulting to '{model_save_base_name}'"
            )

    # Create empty results dictionary if required
    result_fields = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "lr",
        "weight_decay",
        "task",
    ]
    if results is None:
        results = {k: [] for k in result_fields}
    else:
        # check all results lists are the same length
        results_values_lengths = [len(v) for v in results.values()]
        min_results_length = min(results_values_lengths)
        max_results_length = max(results_values_lengths)
        assert_msg = f"[ERROR]: results input contains lists of varying lengths (min: {min_results_length}, max: {max_results_length})."
        assert max_results_length == min_results_length, assert_msg

        # fill any missing fields with None and warn user. Assume epoch started at 1 and fill with 1:max as special case.
        for field in result_fields:
            if field not in results.keys():
                if field == "epoch":
                    results_fill = list(range(1, max_results_length + 1))
                    fill_msg = f"range(1, {max_results_length + 1})"
                else:
                    results_fill = [None] * max_results_length
                    fill_msg = "None"
                results[field] = results_fill
                print(
                    f"[WARNING] Results field '{field}' not in inputted results. Adding to results and padding with {fill_msg}."
                )

    # Set matmul precision (e.g. for TF32 on Ampere+ GPUs)
    torch.set_float32_matmul_precision(float32_matmul_precision)

    # Make sure model on target device and compile if required
    model.to(device)
    if torch_compile:
        model = torch.compile(model)

    # Loop through training and testing steps for a number of epochs
    for _ in tqdm(range(epochs), desc="Training Run"):
        # extract current training config parameters
        epoch_lr = optimizer.param_groups[0]["lr"]
        epoch_weight_decay = optimizer.param_groups[0]["weight_decay"]

        # run training and test steps
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            use_bf16=use_bf16,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            use_bf16=use_bf16,
        )

        # Print out what's happening
        prev_epoch = 0 if len(results["epoch"]) == 0 else max(results["epoch"])
        epoch = prev_epoch + 1
        n_sig_figs = len(
            str(prev_epoch + epochs)
        )  # calc. how much to zero-pad the printing
        task_str = f" | task: {task}" if task is not None else ""
        print(
            f"Epoch: {epoch:0{n_sig_figs}} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"lr: {epoch_lr:.2e} | "
            f"weight_decay: {epoch_weight_decay:.2e}"
            f"{task_str}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["lr"].append(epoch_lr)
        results["weight_decay"].append(epoch_weight_decay)
        results["task"].append(task)
        results["epoch"].append(epoch)

        # Include tensorboard writer updates if required
        if writer:
            # Loss
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "Train": train_loss,
                    "Test": test_loss,
                },
                global_step=epoch,
            )

            # Accuracy
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={
                    "Train": train_acc,
                    "Test": test_acc,
                },
                global_step=epoch,
            )

            # Scheduling
            writer.add_scalars(
                main_tag="Scheduling",
                tag_scalar_dict={
                    "LR": epoch_lr,
                    "Weight_Decay": epoch_weight_decay,
                },
                global_step=epoch,
            )

            # Write values to disk per epoch to avoid loss if loop is interrupted (auto-flushes every 10 writes or 2mins by default)
            writer.flush()

        # add checkpointing model (optional)
        if checkpoint_interval:
            if epoch % checkpoint_interval == 0:
                # check that save path is defined
                if model_save_path:
                    # construct model checkpoint name and save
                    model_name = (
                        model_save_base_name + "_cp" + f"{epoch:0{n_sig_figs}}.pth"
                    )
                    save_checkpoint(
                        target_dir=model_save_path,
                        checkpoint_name=model_name,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        colab_local_path=colab_local_path,
                        scheduler=scheduler,
                        scaler=scaler,
                        test_loss=test_loss,
                        config_file=config_file,
                    )

    # add saving final model (optional)
    if save_final_model:
        if model_save_path:
            model_name = (
                model_save_base_name + "_cp" + f"{epoch:0{n_sig_figs}}" + "_final.pth"
            )
            save_model(
                model=model,
                target_dir=model_save_path,
                model_name=model_name,
            )

    # save model graph
    if (
        writer and False
    ):  # switched off for now as raises control-flow errors when asserts present in model
        try:
            example_batch = next(iter(train_dataloader))[0].to(
                device
            )  # dataloader returns list of [X, y] (each is length batch size)
            model.eval()
            writer.add_graph(model=model, input_to_model=example_batch)
        except Exception as e:
            print(f"[WARNING] Could not add model graph to TensorBoard: {e}")

    # Return the filled results at the end of the epochs and close writer
    if writer:
        writer.close()
    return results


def experiment_sweep(experiment_params: dict):
    """Creates a decorator to run a function over a sweep of experiment parameter combinations.

    Takes a dictionary of experiment parameters and generates all combinations
    using a Cartesian product (i.e. full factorial design). The decorated function
    is then executed once per combination, with results stored in each run's output.

    Args:
        experiment_params (dict): A dictionary where keys are parameter names and
        values are lists of values to sweep over.

    Returns:
        Callable: A decorator that, when applied to a function, runs it across all
        parameter combinations and returns a list of dictionaries containing the
        parameters and associated results.

    Example usage:
        @experiment_sweep({
            "lr": [1e-3, 1e-4],
            "batch_size": [32, 64]
        })
        def run_experiment(lr, batch_size):
            ...

        results = run_experiment()
        # results will be a list of dicts with 'lr', 'batch_size', and 'results' keys
    """

    def experiments_decorator(func):
        # decorator to pass forward all the function attributes
        @functools.wraps(func)

        # wrapper function to run experiments
        def experiment_wrapper(*args, **kwargs) -> List[Dict[str, Any]]:
            keys, values = zip(*experiment_params.items())
            experiments = []

            # iterate through experiments. product function creates cartesian product (a.k.a. full factorial) table of experiment runs
            for run_params in tqdm(list(product(*values)), desc="Experiments"):
                # construct experiment, and combine (|) dictionaries with the experiments params overriding the fixed params if duplicated
                experiment = dict(zip(keys, run_params))
                experiment = kwargs | experiment

                # run experiment function, add results to setup dict, append to output list
                experiment["results"] = func(*args, **experiment)
                experiments.append(experiment)

            # return experiments
            return experiments

        return experiment_wrapper

    return experiments_decorator
