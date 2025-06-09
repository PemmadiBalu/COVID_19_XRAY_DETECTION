🦠 COVID-19 Chest X-ray Detection
This project uses deep learning techniques to detect COVID-19 from chest X-ray images. It leverages convolutional neural networks (CNNs) to classify images into three categories: COVID-19, Normal, and Pneumonia.

📁 Project Structure
COVID19_CHEST_XRAY_DETECTION/
├── dataset/                # X-ray images (COVID, Normal, Pneumonia)
├── models/                 # Saved trained models
├── COVID19_CHEST_XRAY_DETECTION.ipynb  # Main Jupyter notebook
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
🔍 Objective
To develop a deep learning model that can automatically detect COVID-19 infections from chest X-ray scans, helping doctors quickly identify patients who may need further testing or care.

📊 Dataset
Source: Kaggle
Categories:
      COVID-19 Positive
Normal
Viral Pneumonia

🛠️ Technologies Used
1.Python
2.TensorFlow / Keras
3.OpenCV
4.NumPy, Pandas
5.Matplotlib, Seaborn

🧠 Model Architecture
CNN (Convolutional Neural Network)

Layers: Convolutional → ReLU → MaxPooling → Flatten → Dense

Activation: ReLU and Softmax

Loss Function: Categorical Crossentropy

Optimizer: Adam

📈 Results
Class	Precision	Recall	F1-Score
COVID-19	0.97	0.95	0.96
Normal	0.95	0.96	0.95
Pneumonia	0.96	0.97	0.96

Accuracy: 96%

🚀 How to Run
Clone the repository:
- git clone https://github.com/your-username/COVID19_CHEST_XRAY_DETECTION.git
- cd COVID19_CHEST_XRAY_DETECTION
- 
- Install dependencies:
- pip install -r requirements.txt
- Run the notebook:
- jupyter notebook COVID19_CHEST_XRAY_DETECTION.ipynb
- 
📌 Future Work
 @. Improve model accuracy using data augmentation and transfer learning
 @. Deploy the model using Flask or Streamlit
 @. Integrate a web interface for uploading X-ray images

🧑‍💻 Author
- Balu Pemmadi
- B.Tech in Computer Science (AI & Data Science)
- Passionate about AI, deep learning, and real-world healthcare applications

📄 License
- This project is open-source under the MIT License.
