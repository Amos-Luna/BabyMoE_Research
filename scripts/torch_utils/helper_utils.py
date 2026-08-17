import os
import random
from collections import defaultdict

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from IPython.display import display, HTML
from PIL import Image, UnidentifiedImageError
from torchmetrics.classification import Precision, Recall
from torchvision import transforms
from tqdm.auto import tqdm



def dataset_images_per_class(dataset_path):
    """
    Counts the number of image files per class within a dataset directory structure.

    Args:
        dataset_path: The root path of the dataset.
    """
    # Print a message indicating the start of the analysis.
    print(f"Analyzing dataset at: {dataset_path}\n")
    # Define a tuple of valid file extensions for images.
    valid_exts = ('.jpg', '.jpeg', '.png')

    # Define the specific subdirectories to process.
    split_names = ['train', 'test']

    # Use a try block to handle potential file system errors.
    try:
        # Iterate over each defined split name.
        for split_name in split_names:
            # Construct the full path to the current split directory.
            split_path = os.path.join(dataset_path, split_name)

            # Check if the directory exists before proceeding.
            if os.path.isdir(split_path):
                # Print a header for the current split.
                print(f"— {split_name.capitalize()} —")

                # Get a sorted list of all entries in the split directory.
                class_entries = sorted(os.scandir(split_path), key=lambda e: e.name)
                
                # Iterate through each entry within the split directory.
                for class_entry in class_entries:
                    # Check if the entry is a directory (a class folder).
                    if class_entry.is_dir():
                        # Count image files in the current class directory.
                        image_count = sum(1 for file_entry in os.scandir(class_entry.path)
                                         if file_entry.is_file() and file_entry.name.lower().endswith(valid_exts))
                        
                        # Print the count for the current class.
                        print(f"{class_entry.name.capitalize()}: {image_count} images")
                
                # Add a blank line for formatting.
                print()

    # Catch the specific error for a missing directory.
    except FileNotFoundError:
        print(f"Error: The directory '{dataset_path}' was not found.")
    # Catch any other unexpected errors during execution.
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

        

def display_train_images(dataset_path):
    """
    Displays a grid of randomly selected images from a specified training dataset.

    Args:
        dataset_path: The root path to the dataset.
    """
    # Define paths to the different training subdirectories.
    paths = {
        "train/real": os.path.join(dataset_path, 'train', 'real'),
        "train/fake": os.path.join(dataset_path, 'train', 'fake')
    }
    
    # Initialize a list to hold the paths and titles of the images to be plotted.
    images_to_plot = []
    # Define a tuple of valid image file extensions.
    valid_exts = ('.jpg', '.jpeg', '.png')

    # Iterate through each defined subdirectory to gather image paths.
    for title, directory in paths.items():
        # Use a try block to handle potential file system errors.
        try:
            # Use os.scandir for an efficient way to list directory contents.
            with os.scandir(directory) as entries:
                # Create a list of image file paths from the directory.
                image_files = [
                    entry.path for entry in entries 
                    if entry.is_file() and entry.name.lower().endswith(valid_exts)
                ]
            
            # Check if there are enough images to sample from the directory.
            if len(image_files) < 3:
                # Print a warning if the number of images is insufficient.
                print(f"Warning: Not enough images in '{directory}'. Found {len(image_files)}, need 3.")
                # Skip to the next iteration if not enough images are found.
                continue

            # Randomly sample three image paths from the list.
            for img_path in random.sample(image_files, 3):
                # Append the image path and its corresponding title to the list.
                images_to_plot.append((img_path, title))

        # Catch the specific error if the directory does not exist.
        except FileNotFoundError:
            print(f"Error: Directory not found at '{directory}'.")
            # Exit the function if a critical directory is missing.
            return

    # Check if a total of six images were successfully gathered.
    if len(images_to_plot) != 6:
        print("Could not gather enough images to display. Aborting.")
        # Exit the function if the total image count is incorrect.
        return

    # Create a 2x3 grid of subplots for the images.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # Define a fixed size for image thumbnails for faster processing.
    thumbnail_size = (256, 256)

    # Use zip to iterate over both the subplots and the image data simultaneously.
    for ax, (img_path, title) in zip(axes.ravel(), images_to_plot):
        # Use a try block to handle errors during image loading.
        try:
            # Open the image file using the Pillow library.
            with Image.open(img_path) as img:
                # Resize the image to a thumbnail for efficient plotting.
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                
                # Display the image on the current subplot.
                ax.imshow(img)
                # Set the title of the subplot.
                ax.set_title(title)
                # Turn off the axis display for a cleaner look.
                ax.axis('off')

        # Catch specific errors related to an unidentifiable or missing image file.
        except (UnidentifiedImageError, FileNotFoundError):
            # Set an error title for the subplot.
            ax.set_title(f"Error loading image\n{os.path.basename(img_path)}", color='red')
            # Turn off the axis display.
            ax.axis('off')

    # Adjust the spacing between subplots.
    plt.tight_layout()
    # Display the plot.
    plt.show()
    
    
def display_data_loader_contents(data_loader):
    """
    Examines and prints the contents of a data loader, showing details of the first batch.

    Args:
        data_loader: The data loader object to be inspected.
    """
    # Use a try block to handle potential errors during data access.
    try:
        # Print the total number of batches in the data loader.
        print("Length:", len(data_loader))
        # Iterate through the data loader to access batches.
        for batch_idx, (data, labels) in enumerate(data_loader):
            # Print a separator for the current batch.
            print(f"--- Batch {batch_idx + 1} ---")
            # Print the shape of the data tensor.
            print(f"Data shape: {data.shape}")
            # Print the shape of the labels tensor.
            print(f"Labels shape: {labels.shape}")
            # Exit the loop after processing the first batch.
            break
    # Handle the specific case where the data loader is empty.
    except StopIteration:
        print("data loader is empty.")
    # Handle any other unexpected exceptions.
    except Exception as e:
        print(f"An error occurred: {e}")
        


def training_loop(model, train_loader, val_loader, loss_fcn, optmzr, device, num_epochs=3):
    """
    Performs the training and validation loop for a given PyTorch model.
    Saves and returns the model with the highest validation accuracy.

    Args:
        model: The model to be trained.
        train_loader: The data loader for the training dataset.
        val_loader: The data loader for the validation dataset.
        loss_fcn: The loss function to compute the training loss.
        optmzr: The optimizer to update model parameters.
        device: The device (CPU or CUDA) on which the model and data will be processed.
        num_epochs: The total number of epochs for training.

    Returns:
        The trained model object with the weights that achieved the highest validation accuracy.
    """
    # Create the directory to save the best model if it doesn't exist.
    save_dir = "./best_model_saved/"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # Move the model to the specified computing device.
    model.to(device)
    # Assign the provided loss function.
    loss_function = loss_fcn
    # Assign the provided optimizer.
    optimizer = optmzr

    # Determine the number of classes from the model's final output layer.
    num_classes = model.classifier[-1].out_features

    # Initialize accuracy, precision and recall metrics for validation.
    val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device)
    val_precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    val_recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    
    # Initialize variables to track the best performance.
    best_val_accuracy = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0

    # Begin the main training and validation loop for a number of epochs.
    for epoch in range(num_epochs):
        # Set the model to training mode.
        model.train()
        # Initialize a variable to accumulate the training loss.
        running_loss = 0.0
        # Initialize a counter for correctly classified training samples.
        total_train_correct = 0
        # Initialize a counter for total training samples.
        total_train_samples = 0

        # Create a progress bar for the training batches.
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", unit="batch")
        # Iterate over batches from the training data loader.
        for images, labels in train_progress_bar:
            # Move the images and labels to the designated device.
            images, labels = images.to(device), labels.to(device)
            # Reset the gradients of all optimized tensors to zero.
            optimizer.zero_grad()
            # Perform a forward pass to get model outputs.
            outputs = model(images)
            # Compute the loss between the outputs and the true labels.
            loss = loss_function(outputs, labels)
            # Perform backpropagation to compute gradients.
            loss.backward()
            # Update the model's weights using the optimizer.
            optimizer.step()

            # Accumulate the loss and sample counts.
            running_loss += loss.item() * labels.size(0)
            # Get the predicted class with the highest probability.
            _, predicted = torch.max(outputs, dim=1)
            # Update the counter for correct predictions.
            total_train_correct += (predicted == labels).sum().item()
            # Update the total count of samples processed.
            total_train_samples += labels.size(0)
            
            # Calculate the average loss for the current epoch.
            epoch_loss = running_loss / total_train_samples
            # Calculate the accuracy for the current epoch.
            epoch_acc = 100 * total_train_correct / total_train_samples
            # Update the progress bar with real time loss and accuracy.
            train_progress_bar.set_postfix(loss=f"{epoch_loss:.4f}", accuracy=f"{epoch_acc:.2f}%")

        # Begin the validation phase.
        # Set the model to evaluation mode.
        model.eval()
        # Initialize a counter for total validation samples.
        total_val_samples = 0
        # Initialize a variable to accumulate validation loss.
        val_loss = 0.0
        
        # Reset the validation metric objects for the new epoch.
        val_accuracy_metric.reset()
        val_precision_metric.reset()
        val_recall_metric.reset()

        # Disable gradient calculations for efficiency during validation.
        with torch.no_grad():
            # Create a progress bar for the validation batches.
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validating", unit="batch")
            # Iterate over batches from the validation data loader.
            for images, labels in val_progress_bar:
                # Move the images and labels to the designated device.
                images, labels = images.to(device), labels.to(device)
                # Perform a forward pass.
                outputs = model(images)
                # Compute the loss.
                loss = loss_function(outputs, labels)
                # Accumulate the validation loss.
                val_loss += loss.item() * labels.size(0)
                # Get the predicted class.
                _, predicted = torch.max(outputs, dim=1)
                # Update the total count of validation samples.
                total_val_samples += labels.size(0)

                # Update the metric objects with the batch predictions and labels.
                val_accuracy_metric.update(predicted, labels)
                val_precision_metric.update(predicted, labels)
                val_recall_metric.update(predicted, labels)
                
                # Update the progress bar with the current accuracy.
                val_progress_bar.set_postfix(
                    accuracy=f"{100 * val_accuracy_metric.compute():.2f}%"
                )

        # Calculate the average validation loss for the epoch.
        avg_val_loss = val_loss / total_val_samples
        
        # Compute the final metric values for the entire epoch.
        final_val_acc = val_accuracy_metric.compute()
        final_val_precision = val_precision_metric.compute()
        final_val_recall = val_recall_metric.compute()
        
        # Print a summary of the validation results for the epoch.
        print(f'Val Loss (Avg): {avg_val_loss:.4f}, Val Accuracy: {final_val_acc * 100:.2f}%\n')

        # Check if the current model is the best one and save it.
        if final_val_acc > best_val_accuracy:
            best_val_accuracy = final_val_acc
            best_val_precision = final_val_precision
            best_val_recall = final_val_recall
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with Val Accuracy: {best_val_accuracy * 100:.2f}%\n")


    # Print a message indicating the completion of training.
    print("\nTraining ended. Best trained model returned.")
    print(f"Best Val Accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Best Val Precision: {best_val_precision:.4f}")
    print(f"Best Val Recall: {best_val_recall:.4f}\n")
    
    # Load the best model weights before returning.
    model.load_state_dict(torch.load(best_model_path))
    
    # Return the best trained model.
    return model



def visualize_predictions(model, data_loader, device, class_names):
    """
    Creates a visual grid of images showing their true and predicted class labels.

    Args:
        model: The trained model to be used for making predictions.
        data_loader: An object that loads the dataset in batches.
        device: The computing device on which to run the model.
        class_names: A list of class names that correspond to the labels.
    """

    # Set the model to evaluation mode to disable layers like dropout.
    model.eval()
    # Move the model to the specified device.
    model.to(device)

    # Define the pre-calculated dataset mean.
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    # Define the pre-calculated dataset standard deviation.
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    # Initialize lists to store all images and labels for each class.
    all_class_0_data = []
    all_class_1_data = []

    # Disable gradient calculations for faster inference.
    with torch.no_grad():
        # Iterate through the entire data loader to collect all images and their labels.
        # This is necessary for true randomness, as it ensures all images are in the pool for sampling.
        for images, labels in data_loader:
            # Move the image and label data to the specified device.
            images, labels = images.to(device), labels.to(device)

            # Find and collect all images belonging to class 0 from the batch.
            indices_class_0 = (labels == 0).nonzero(as_tuple=True)[0]
            for idx in indices_class_0:
                all_class_0_data.append((images[idx], labels[idx]))
            
            # Find and collect all images belonging to class 1 from the batch.
            indices_class_1 = (labels == 1).nonzero(as_tuple=True)[0]
            for idx in indices_class_1:
                all_class_1_data.append((images[idx], labels[idx]))
    
    # Randomly sample three images from the complete list of images for each class.
    sampled_class_0_data = random.sample(all_class_0_data, 3)
    # Randomly sample three images from the complete list of images for each class.
    sampled_class_1_data = random.sample(all_class_1_data, 3)
    
    # Combine the sampled images and labels from both classes.
    all_data = sampled_class_0_data + sampled_class_1_data
    all_images = torch.stack([item[0] for item in all_data])
    all_labels = torch.stack([item[1] for item in all_data])

    # Pass the collected images through the model to get the outputs.
    outputs = model(all_images)
    # Get the predicted class with the highest probability.
    _, predicted = torch.max(outputs, 1)

    # Move the tensors back to the CPU for plotting.
    images = all_images.cpu()
    # Move the true labels back to the CPU.
    labels = all_labels.cpu()
    # Move the predicted labels back to the CPU.
    predicted = predicted.cpu()

    # Create a new figure with a specified size.
    plt.figure(figsize=(10, 10))
    # Loop through a fixed number of images to display.
    for i in range(6):
        # Create a subplot within the figure.
        plt.subplot(2, 3, i + 1)
        # Change the tensor's dimension order for image display and convert to a numpy array.
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize the image using the dataset mean and standard deviation.
        img = imagenet_std.numpy() * img + imagenet_mean.numpy()
        # Clip the image pixel values to ensure they are within a valid range.
        img = np.clip(img, 0, 1)
        # Display the image.
        plt.imshow(img)

        # Retrieve the true class name for the current image.
        true_label = class_names[labels[i]]
        # Retrieve the predicted class name for the current image.
        pred_label = class_names[predicted[i]]

        # Set the title color to green for a correct prediction and red otherwise.
        color = "green" if true_label == pred_label else "red"
        # Set the title of the subplot with true and predicted labels.
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=12)
        # Hide the axis.
        plt.axis("off")
    # Adjust the layout to prevent titles and labels from overlapping.
    plt.tight_layout(h_pad=0.5)
    # Display the final plot.
    plt.show()
    

def upload_jpg_widget():
    """
    Creates and displays a file upload widget for JPG images.

    This function is designed for Jupyter or IPython environments. It provides an
    interface for uploading a single JPG file. The function validates that the
    uploaded file has a '.jpg' extension and is no larger than 5MB. Valid files
    are saved to a local './images' directory. Feedback, including success or
    error messages, is displayed directly in the output.
    """
    # Define the target directory for uploaded images
    output_image_folder = "./images"
    # Ensure the target directory exists, creating it if necessary
    os.makedirs(output_image_folder, exist_ok=True)

    # Create the file upload widget with specific constraints
    uploader = widgets.FileUpload(
        accept='.jpg',
        multiple=False,
        description='Upload JPG (Max 5MB)'
    )

    # Create an output widget to display status messages
    output_area = widgets.Output()

    # Define the callback function to handle file uploads
    def on_file_uploaded(change):
        """
        Handles the logic when a file is uploaded via the widget.
        
        This function is triggered on a change to the uploader's value. It
        validates and saves the uploaded file.
        """
        # Retrieve the newly uploaded file data from the change event
        current_uploaded_value_tuple = change['new']

        # Exit if the callback was triggered by the value being cleared
        if not current_uploaded_value_tuple:
            return

        # Use the output area to display messages for this upload attempt
        with output_area:
            # Clear any messages from a previous upload
            output_area.clear_output()

            # Extract file details from the uploaded data
            file_data_dict = current_uploaded_value_tuple[0]
            filename = file_data_dict['name']
            file_content = file_data_dict['content']

            # Validate that the file has a '.jpg' extension
            if not filename.lower().endswith('.jpg'):
                # Construct and display a format error message
                error_msg_format = (
                    f"<p style='color:red;'>Error: Please upload a file with a ‘.jpg’ format. "
                    f"You uploaded: '{filename}'</p>"
                )
                display(HTML(error_msg_format))
                # Clear the invalid file from the widget
                uploader.value = ()
                return

            # Validate the file size against the 5MB limit
            file_size_bytes = len(file_content)
            max_size_bytes = 5 * 1024 * 1024

            if file_size_bytes > max_size_bytes:
                # Calculate size in MB for the error message
                file_size_mb = file_size_bytes / (1024 * 1024)
                # Construct and display a size error message
                error_msg_size = (
                    f"<p style='color:red;'>Error: File '{filename}' is too large ({file_size_mb:.2f} MB). "
                    f"Please upload a file less than or equal to 5 MB.</p>"
                )
                display(HTML(error_msg_size))
                # Clear the oversized file from the widget
                uploader.value = ()
                return

            # Attempt to save the validated file
            try:
                # Construct the full path for the new file
                save_path = os.path.join(output_image_folder, filename)

                # Write the file content in binary mode
                with open(save_path, 'wb') as f:
                    f.write(file_content)

                # Get the string representation of the path for the success message
                python_code_path = repr(save_path)

                # Construct and display the success message
                success_message = f"""
                <p style='color:green;'>File successfully uploaded!</p>
                <p>Please use the path as <code>image_path = {python_code_path}</code></p>
                """
                display(HTML(success_message))

            except Exception as e:
                # Display an error message if saving fails
                error_msg_save = f"<p style='color:red;'>Error saving file '{filename}': {e}</p>"
                display(HTML(error_msg_save))
            finally:
                # Clear the uploader's value to reset it for the next upload
                uploader.value = ()

    # Register the callback function to observe the 'value' property of the uploader
    uploader.observe(on_file_uploaded, names='value')

    # Display the uploader widget and the associated output area
    display(uploader)
    display(output_area)



def make_predictions(model, image_path, device, class_names):
    """
    Loads a single image, applies transformations, makes a prediction,
    and displays the image with its predicted class label.

    Args:
        model (torch.nn.Module): The trained model for making predictions.
        image_path (str): The file path to the input .jpg image.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        class_names (list): A list of class names corresponding to the labels.
    """
    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    # Define the mean and std for normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Define the transformation pipeline
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 1. Load and transform the image
    try:
        # Open the image and ensure it's in RGB format
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return
        
    image_tensor = val_transform(image)
    
    # Add a batch dimension (models expect a batch of images)
    # The shape changes from [C, H, W] to [1, C, H, W]
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # 2. Make a prediction
    with torch.no_grad():
        # Get the model's raw output (logits)
        outputs = model(input_tensor)
        # Get the index of the highest score, which is our predicted class
        _, predicted_idx = torch.max(outputs, 1)

    # Get the predicted class name
    pred_label = class_names[predicted_idx.item()]

    # 3. Display the image and its prediction
    plt.figure(figsize=(6, 6))
    
    # To display the image correctly, we need to denormalize the tensor
    img_to_display = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_to_display = std.numpy() * img_to_display + mean.numpy()
    img_to_display = np.clip(img_to_display, 0, 1) # Clip values to be between 0 and 1

    plt.imshow(img_to_display)
    plt.title(f"Prediction: {pred_label}", fontsize=14)
    plt.axis("off")
    plt.show()