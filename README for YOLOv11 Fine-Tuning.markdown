# YOLOv11 Fine-Tuning on Agricultural Pests Dataset

## Description
This repository contains a Jupyter notebook that demonstrates how to fine-tune the YOLOv11 object detection model on the Agricultural Pests Dataset using Google Colab. The goal is to develop a model that can accurately detect and classify different types of agricultural pests, such as moths, slugs, snails, wasps, and weevils, which can be useful for precision agriculture and pest management.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Fine-Tuning Steps](#fine-tuning-steps)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
To run this notebook, you need to have the following installed:
- Python 3.x
- Jupyter Notebook
- PyTorch
- Ultralytics YOLO library

You can install the Ultralytics YOLO and other libraries using pip:
```bash
pip install ultralytics kaggle torch torchvision opencv-python matplotlib
```

Since the notebook is designed to run on Google Colab, you can open it directly in Colab by uploading it to your Colab environment.

**Note**: Ensure you have a Google account and enable GPU acceleration in Colab (Edit > Notebook Settings > Hardware Accelerator > GPU) for faster training.

## Dataset
The dataset used in this project is the Agricultural Pests Dataset, which can be downloaded from Kaggle. The dataset includes images of various pests such as moths, slugs, snails, wasps, and weevils, along with their annotations in the form of bounding boxes.

To use the dataset:
1. Create a Kaggle account if you don't have one.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/agricultural-pests-dataset).
3. Upload the dataset to your Google Drive.
4. In the notebook, mount your Google Drive and access the dataset.

The dataset should be organized into `train`, `val`, and `test` folders, with images and annotations in the appropriate formats. A YAML file is required to define the dataset paths and classes, for example:

```yaml
path: /content/pest_data_yolo_classification/
train: train
val: val
test: test
nc: 12
names: ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']
```

## Model
We use the pre-trained YOLOv11 model provided by Ultralytics. YOLOv11 is a state-of-the-art object detection model known for its speed and accuracy. The model can be loaded using the Ultralytics library:

```python
# Step 5: Train YOLOv11 classification model
from ultralytics import YOLO
import os

# Verify dataset structure
dataset_yaml = '/content/pest_data_yolo_classification/'
if not os.path.exists(dataset_yaml):
    raise FileNotFoundError(f"Dataset YAML file not found at {dataset_yaml}")

# Check if the dataset contains images
def verify_dataset_split(split_path):
    if not os.path.exists(split_path):
        return False
    class_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    if not class_dirs:
        return False
    for class_dir in class_dirs:
        class_path = os.path.join(split_path, class_dir)
        if not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(class_path)):
            return False
    return True

for split in ['train', 'val']:
    split_path = os.path.join('/content/pest_data_yolo_classification', split)
    if not verify_dataset_split(split_path):
        raise ValueError(f"Dataset split {split} is empty or improperly structured at {split_path}")

# Load YOLOv11 classification model
try:
    # Load YOLOv11 classification model (automatically downloads if not found)
    model = YOLO('yolo11s-cls.pt')  # Nano version for faster training
    print("YOLOv11 model loaded successfully. Architecture summary:")
    model.info()  # Print model summary
except Exception as e:
    raise RuntimeError(f"Failed to load YOLOv11 model: {e}")

# Training configuration
try:
    results = model.train(
        data=dataset_yaml,
        epochs=25,
        imgsz=224,
        task='classify',
        batch=64,
        patience=5,
        device='0',
        workers=4,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        seed=42,
        name='yolo11s_pest_classification',  # Changed to reflect YOLOv11
        exist_ok=True
    )
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {results.save_dir}")
    print(f"Top-1 Accuracy: {results.metrics.top1:.2f}%")
    print(f"Top-5 Accuracy: {results.metrics.top5:.2f}%")
    
except Exception as e:
    print(f"\nTraining failed with error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Verify the dataset YAML file exists at the correct path")
    print("2. Check that images exist in train/ and val/ directories with proper class subfolders")
    print("3. Ensure images are valid (try opening some manually)")
    print("4. Try reducing batch size if you get CUDA memory errors")
    print("5. Try running with device='cpu' if GPU is not available")
    print("6. Check disk space (training creates many checkpoint files)")
    
    
```

## Fine-Tuning Steps
The fine-tuning process involves the following steps:

1. **Data Preparation**:
   - Load the dataset from Google Drive.
   - Split the data into training, validation, and test sets.
   - Create a YAML file that defines the dataset paths and classes.

2. **Model Configuration**:
   - Load the pre-trained YOLOv11 model.
   - Define the custom classes for pest detection (e.g., moth, slug, snail, wasp, weevil).
   - Set hyperparameters such as learning rate, batch size, and number of epochs.

3. **Training**:
   - Train the model on the training set using the `train` method from the Ultralytics library.
   - Validate the model on the validation set after each epoch.
   - Save the best-performing model based on validation metrics.

   Example training command:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11s-cls.pt')
    results = model.train(
        data=dataset_yaml,
        epochs=25,
        imgsz=224,
        task='classify',
        batch=64,
        patience=5,
        device='0',
        workers=4,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        seed=42,
        name='yolo11s_pest_classification',  # Changed to reflect YOLOv11
        exist_ok=True
    )
   ```

4. **Evaluation**:
   - Evaluate the final model on the test set.
   - Calculate metrics such as mean Average Precision (mAP), precision, recall, and F1-score.
   - Visualize the results, including confusion matrices and class accuracy plots.

5. **Inference**:
   - Use the fine-tuned model to detect pests in new images.
   - Visualize the detections.

   

## Results
After fine-tuning, the model is expected to achieve high accuracy in detecting agricultural pests, with metrics such as mean Average Precision (mAP) typically reported. The notebook includes visualizations such as confusion matrices and class accuracy plots to provide insights into the model's performance. Specific metrics are not available without the notebook's output, but similar projects report mAP values around 0.8-0.9 for well-prepared datasets.

## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes. We welcome improvements, bug fixes, and additional features.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements
- Ultralytics for providing the YOLOv11 model and library.
- Kaggle for hosting the Agricultural Pests Dataset.
- The original creators of the dataset and the YOLOv11 model.

---

**Note**: This notebook is designed to run on Google Colab. You can open it directly in Colab by uploading it to your Colab environment.