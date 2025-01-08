from google.colab import drive
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import pandas as pd
import os

# Mount Google Drive
drive.mount('/content/drive')

# Function to load the dataset from Google Drive
def load_dataset(file_path):
    """
    Load the dataset from the given file path.

    Args:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

# Function to extract features from the food image using ResNet-50
def extract_features(img):
    """
    Extract features from an image using a pre-trained ResNet-50 model.

    Args:
        img (PIL.Image.Image): The input image.

    Returns:
        torch.Tensor: The extracted features.
    """
    model = models.resnet50(pretrained=True)  # Load ResNet-50 with ImageNet weights
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(img)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features

# Function to match the image name with the dataset and generate a recipe
def image_to_recipe(img_name, df):
    """
    Perform image-to-recipe generation by matching the food item from the dataset and generating a recipe.

    Args:
        img_name (str): The name of the uploaded image file.
        df (pd.DataFrame): The dataset containing food items and ingredients.

    Returns:
        tuple: The matched food item name, ingredients, and instructions, or an error message.
    """
    try:
        # Hardcoded image name as per your requirement
        img_name = "miso-butter-roast-chicken-acorn-squash-panzanella.jpg"
        
        # Remove file extension from image name
        img_name_no_ext = img_name.rsplit('.', 1)[0]
        
        print(f"Searching for image: {img_name_no_ext}")

        # Check if the image name (without extension) exists in the dataset
        if img_name_no_ext in df['Image_Name'].values:
            matched_food_item = df[df['Image_Name'] == img_name_no_ext].iloc[0]
            food_item_name = matched_food_item['Title']
            ingredients = matched_food_item['Cleaned_Ingredients']
            instructions = matched_food_item['Instructions']
            
            print(f"Found matched recipe: {food_item_name}")
            return food_item_name, ingredients, instructions
        else:
            print("Image name not found in the dataset.")
            return None, None, "Image name not found in the dataset"
    except Exception as e:
        print(f"Error: {e}")
        return None, None, f"Error generating recipe: {e}"

# Function to display the image and generate recipe
def main():
    """
    Main function to upload an image and generate a recipe based on the uploaded image.
    """
    # Path to the dataset and image on Google Drive
    dataset_path = '/content/drive/MyDrive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
    image_path = '/content/drive/MyDrive/miso-butter-roast-chicken-acorn-squash-panzanella.jpg'
    # -chickpea-pancakes-with-leeks-squash-and-yogurt-51260630.jpg
    #-candy-corn-frozen-citrus-cream-pops-368770.jpg
    #-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813.jpg

    # Load the dataset
    df = load_dataset(dataset_path)

    # Print the first few rows of the dataframe to verify the columns
    print("Dataset columns:")
    print(df.columns)
    print(df.head())

    # Open the image
    img = Image.open(image_path)

    # Display the uploaded image
    img.show()

    # Get the image name (without extension)
    img_name = os.path.basename(image_path)

    # Generate recipe using the hardcoded image name
    food_item_name, ingredients, instructions = image_to_recipe(img_name, df)

    # Display generated recipe
    if food_item_name:
        print('Matched Food Item:')
        print(food_item_name)
        
        print('\nIngredients:')
        print(ingredients)

        print('\nInstructions:')
        print(instructions)
    else:
        print('Failed to generate recipe:')
        print(instructions)  # Display the error message

# Run the main function
if __name__ == '__main__':
    main()
