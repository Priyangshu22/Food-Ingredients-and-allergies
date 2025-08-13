🍽️ Food Ingredients and Allergens Prediction

This project uses **Machine Learning** and **Image Processing** to identify food items from images and predict whether they contain allergens specified by the user.  
It is designed to help people with food allergies make safer eating choices.


📌 Overview
- **Goal**: Identify food items from an image and detect allergenic ingredients.
- **Approach**: 
  - Use **Convolutional Neural Networks (CNNs)** for food recognition.
  - Retrieve and analyze nutritional data from online sources.
  - Predict allergen presence based on user-specified allergies.
- **Accuracy**: Achieved up to **85% recognition accuracy** with CNN-based models.
- **Applications**: Healthcare, diet planning, food safety monitoring.

 🗂 Dataset
- **Name**: Food Allergens Dataset ([Kaggle Link](https://www.kaggle.com/datasets/uom190346a/food-ingredients-and-allergens))
- **Size**: ~400 records
- **Attributes**:
  - **Food Item**: Name of the dish or product.
  - **Ingredients**: List of ingredients.
  - **Allergens**: Known allergens in the food item.
  - **Prediction**: Binary classification — contains allergens / does not contain allergens.



 ⚙️ Project Workflow
1. **Data Collection** — Gather food images and allergen info from datasets & online sources.
2. **Data Preprocessing** — Image formatting, ingredient cleaning, label encoding.
3. **Model Training** — Train CNN for image classification (Food-101 + custom dataset).
4. **Ingredient Extraction** — Retrieve ingredients & nutritional facts via web scraping.
5. **Allergen Prediction** — Match extracted ingredients with user-defined allergy list.
6. **Output** — Display allergen warning if unsafe.



🛠 Tech Stack
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow/Keras
- **ML Models**: CNN for image recognition, Neural Networks for ingredient prediction
- **Other Tools**: Web scraping (BeautifulSoup/Requests), Kaggle datasets



 📊 Model Architecture
- **CNN Layers**:
  - Convolution → Activation → Pooling
  - Flattening
  - Fully Connected Layers
- **Output**: Probability distribution over possible food classes.



💻 Installation & Usage
# Clone the repository
git clone https://github.com/<your-username>/food-allergens-prediction.git
cd food-allergens-prediction

# Install dependencies
pip install -r requirements.txt

# Run the training script
python train_model.py

# Predict allergens from an image
python predict.py --image test.jpg --allergy "peanuts,milk"


<img src="images/screenshot1" width="500">
![Homepage Screenshot](/Screenshot 2025-08-14 002820.png")
