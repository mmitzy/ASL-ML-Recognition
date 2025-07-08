# ASL Recognition Project

This project focuses on recognizing American Sign Language (ASL) gestures using a the ASL Alphabet dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
and a variety of machine learning (ML) models. The goal is to explore the data through visualization, reason through our initial preconceptions, and compare several ML approaches to determine the most effective method for ASL recognition.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Visualization](#data-visualization)
- [Preconceptions and Reasoning](#preconceptions-and-reasoning)
- [Machine Learning Models](#machine-learning-models)
- [Getting Started](#getting-started)
- [Results and Evaluation](#results-and-evaluation)
- [Demo](#Demo)
- [Conclusions](#conclusions)
- [Important Note](#important-note)


## Project Overview

The project involves:
- **Dataset Selection:** Choosing a dataset that represents various ASL gestures.
- **Dataset Preprocessing:** Manually dealing with data imbalance and making sure the train/test split is ideal.
- **Data Visualization:** Exploring the dataset through visualizations to understand its distribution, potential biases, and underlying patterns.
- **Hypothesis Formation:** Documenting preconceptions about the data and the challenges in recognizing similar ASL gestures.
- **Model Implementation:** Using multiple machine learning models—including convolutional neural networks (CNNs), support vector machines (SVMs), and ensemble methods—to analyze and classify the ASL images.
- **Evaluation:** Comparing model performance to select the most effective approach for ASL recognition.

## Dataset

The dataset consists of images depicting different ASL signs. Each image is labeled according to the corresponding alphabet or gesture. Detailed information about the dataset, including sample images and class distributions, can be found in the link provided above.

## Project Structure

In this project I included several ML models to test my hypothesis: "By using data augmentation, deep learning and feature extracting, we can improve the issue of ASL Alphabet recognition - similar hand gestures".
The project was devided to 3 main proccesses:
- **Invetigation Phase:** Finding the best possible dataset, thinking of a hypothesis and planning my way of work.
- **Coding Phase:** Writing the ML models and tuning the hyperparameters that will get the most out of the model.
- **Conclusion Phase:** Understanding where I failed and where I found success, looking for a potential answer for the hypothesis.

## Data Visualization

Visual exploration of the dataset was an integral first step. The visualizations include:
- **Class Distribution:** Bar charts showing the frequency of each ASL sign.
- **Sample Images:** A gallery of sample images per category to inspect quality and variability.

These visualizations help uncover potential challenges such as imbalanced classes or high intra-class variability.

## Preconceptions and Reasoning

Before diving into model training, several preconceptions were considered:
- **Image Variability:** ASL gestures may vary widely in appearance due to lighting, background, and signer differences.
- **Model Performance:** Different ML models might have varying levels of performance, especially given the complexity of image data.
- **Feature Discrimination:** Some ASL signs are visually similar, posing a challenge for classifiers.

These insights guided the exploratory data analysis and influenced the selection and tuning of machine learning models.

## Machine Learning Models

Multiple approaches were experimented with:
- **Convolutional Neural Networks (CNNs):** To leverage their strength in capturing spatial hierarchies in image data.
- **Feedforward Neural Network (FFNN):** To present the benefits of using neural networks.
- **Logistic Regression:** To give an example of the problems we might face when using the wrong model for the task.
- **Support Vector Machine (SVM):** To demonstrate a strong classical method for high-dimensional classification, despite computational limitations on large image data.
- **Random Forest:** To illustrate how ensemble methods can perform well even without deep learning, by combining many decision trees.
- **K-Nearest Neighbors (KNN):** To highlight a simple, intuitive method that can work surprisingly well but struggles with scalability and high-dimensional spaces.

Each model’s architecture, training process, and evaluation metrics are detailed in the scripts under the evaluation and it's own directory.

## Getting Started

### Prerequisites


**Models:**
- Python 3.x
- Jupyter Notebook (for running the notebooks)
- Required Python packages (listed in `requirements.txt`)
  
**Demo:**
- Required Python packages (listed in 'requirements_demo.txt')

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mmitzy/ASL-Recognition.git
   cd ASL-Recognition
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt -r requirements_demo.txt
   ```

## Results and Evaluation

Each model was evaluated using metrics such as accuracy, precision, recall, and F1-score. Detailed results are documented in the pictures below.
- Model strengths and weaknesses.
- The impact of data pre-processing and augmentation.
- Comparative performance across different classifiers.

## CNN
![image](https://github.com/user-attachments/assets/1b8b8358-b9ad-4fad-a1bb-37bbac560ab7)
![image](https://github.com/user-attachments/assets/da4072b6-e5f4-44e4-b307-e523242c6434)
![image](https://github.com/user-attachments/assets/a2d4271f-ca7d-4135-8553-fc0e20f65567)

## FFNN
![image](https://github.com/user-attachments/assets/9541f02e-eb38-40cd-95b5-09bec4a3aad5)
![image](https://github.com/user-attachments/assets/920a283d-58e7-4405-a0b1-218c25706d79)

## Logistic Regression
![image](https://github.com/user-attachments/assets/f33b1662-2399-4f00-a393-33adc6372b62)
![image](https://github.com/user-attachments/assets/e116a3f4-76a1-4e1d-9741-8d665b028744)

## KNN
![image](https://github.com/user-attachments/assets/564a2098-293d-423a-affa-21dc81223343)
![image](https://github.com/user-attachments/assets/3caa0e38-61f4-442f-946e-41c8c1603880)

## SVM
![image](https://github.com/user-attachments/assets/5aa1fc08-d14f-49d4-94d7-4c4304d72f63)
![image](https://github.com/user-attachments/assets/992a8c36-1276-43ed-9f8d-926b88ad6859)

## Random Forest
![image](https://github.com/user-attachments/assets/c3fed4ff-1c94-4272-a9ad-2535367d8d89)
![image](https://github.com/user-attachments/assets/aaba1c36-042f-4e31-9517-2d76f9f96000)

## Visualization Testing
![image](https://github.com/user-attachments/assets/421a8e78-a55f-466a-9c3f-e60e176b05b3)
![image](https://github.com/user-attachments/assets/e3b6972f-7302-4644-af55-a4cee9752dc9)
![image](https://github.com/user-attachments/assets/1b77a3e8-2a64-4526-b52a-bb845cc9a15d)
![image](https://github.com/user-attachments/assets/73b59b78-9f6b-4025-a140-afa455f43be6)


## Bonus

In order to truly test the model in action, we made a demo where you can try to make complete sentences using the ASL.
- Model uses ensemble learning in order to split the weight between models.
- Can input images manually
- Can input images using a webcam

![image](https://github.com/user-attachments/assets/05ca0413-0d56-481e-8571-ccfa2927a7c3)
![image](https://github.com/user-attachments/assets/99e4ecc5-aad2-4eed-9cb1-a8099d1322d7)
![image](https://github.com/user-attachments/assets/b36ddccb-5301-4324-b676-6c3dc9a2eb59)
![image](https://github.com/user-attachments/assets/df5eace5-8ec1-460e-a2c0-0fdbdf63fdad)

## Conclusions

After improving from the other models and achieving the incredible accuracy of 99% using CNN, we understood that our hypothesis is probably correct. Of course 1 example isn't enough to determine such a big question, but due to the usage of a very big dataset (consisting of 84k+ images) we believe that it's very likely that we are right.


## Important Note

The models are all named in the same format: 'modelname'.py
All other files including any combination of the model name is related to the training save files.
