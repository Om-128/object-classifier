# 🧠 CIFAR-10 Object Classifier

A simple **Convolutional Neural Network (CNN)** project to classify images from the **CIFAR-10** dataset into 10 object categories.  
Built to understand how to structure an **AI/ML project modularly** — including data loading, preprocessing, model training, and prediction pipelines.

---

## 📂 Dataset
**CIFAR-10** contains **60,000 color images (32x32 pixels)** in **10 classes**:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

---

## ⚙️ Features
- 🧩 Modular project structure  
- 🧼 Data preprocessing  
- 🧠 CNN model training  
- 📈 Model evaluation and prediction pipeline  

---

## 🚀 How to Run
```bash

### 1️⃣ Setup Environment
    - git clone https://github.com/<your-username>/object-classifier.git
    - cd object-classifier
    - python -m venv venv
    - venv\Scripts\activate      # for Windows
    - pip install -r requirements.txt

---

### 2️⃣ Train the Model
python -m src.pipeline.train_pipeline
    - This will train the model and save:
    - model.h5 → trained model
    - preprocessor.pkl → preprocessing object

---

### 3️⃣ Make Predictions
python -m src.pipeline.predict_pipeline
    - Make sure you have an image file like sample.jpg in the project folder.
    - Example Output:
        - Predicted class: 🐶 dog
        - Confidence: 87.34%