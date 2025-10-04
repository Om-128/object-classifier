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

## ⚠️ Challenges & Observations

- When using random internet images, predictions were often incorrect.

- Predictions improved significantly (60–70% accuracy) when using images from this GitHub repo:
CIFAR-10 images

- The model currently performs well on dataset-like images but struggles with arbitrary real-world images.

---

## 🔮 Future Improvements

- Implement data augmentation to improve generalization on unseen images.

- Train with a deeper CNN architecture for better feature extraction.

- Fine-tune hyperparameters for higher accuracy.

- Expand preprocessing to handle diverse real-world images for robust prediction.

- Possibly integrate transfer learning using pretrained models for better performance on small datasets.

