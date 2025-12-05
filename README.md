# 🧠 SmartML - Automated Machine Learning Platform

SmartML is a powerful, user-friendly platform designed to democratize machine learning. It automates the entire ML lifecycle—from data preprocessing and model training to evaluation and deployment—allowing users to build and deploy robust models without writing a single line of code.

## ✨ Key Features

- **🔐 User Authentication**: Secure registration and login system using JWT and bcrypt.
- **📂 Dataset Management**: Easy CSV upload with automatic column detection and validation.
- **🛠️ Robust Preprocessing**:
    - Automatic missing value imputation (Mean/Mode/KNN).
    - Intelligent outlier detection and removal.
    - Categorical encoding (One-Hot, Target, Frequency).
    - Feature scaling and high-correlation removal.
    - Automated PCA for dimensionality reduction if the dataset has more features.
- **🤖 AutoML Engine**:
    - Supports both **Classification** and **Regression** tasks.
    - Trains multiple algorithms (Random Forest, XGBoost, Linear/Logistic Regression, SVM, etc.).
    - Hyperparameter tuning using Grid/Random Search.
    - Automatic model selection based on optimal metrics.
- **📦 One-Click Deployment**:
    - Exports a self-contained **Bundle** (Zip file).
    - Includes a ready-to-use **Streamlit App** (`app.py`).
    - Contains the trained model and a portable, dependency-free preprocessor.
    - Generates a `requirements.txt` for easy setup.
- **📊 Model Insights**: Detailed performance metrics and model explanations.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- MongoDB (running locally or cloud URI)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/SmartML.git
    cd SmartML
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```env
    MONGO_DB_URI=mongodb://localhost:27017/
    SECRET_KEY=your_secret_key_here
    ALGORITHM=HS256
    ```

### Running the Application

Start the FastAPI backend server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
Access the automatic API docs at `http://127.0.0.1:8000/docs`.

## 🏗️ Project Structure

```
SmartML/
├── components/             # Core ML logic
│   └── preprocessing.py    # Sklearn-compatible preprocessor
├── user_section/           # User-specific logic
│   ├── training/           # Training scripts & bundle exporter
│   └── prediction/         # Prediction pipelines
├── pydantic_models/        # Data validation models
├── utils/                  # Helper functions (JWT, etc.)
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🎁 Deployment Bundles

When you export a model, SmartML generates a zip file containing:

-   `model.pkl`: The trained estimator.
-   `preprocessor.pkl`: A lightweight, inference-ready preprocessor.
-   `app.py`: A Streamlit web interface for immediate inference.
-   `requirements.txt`: Dependencies for the bundled app.

To run a downloaded bundle:
1. Unzip the folder.
2. Run `pip install -r requirements.txt`.
3. Run `streamlit run app.py`.

## 🛠️ Technologies Used

-   **Backend**: FastAPI, Python
-   **Database**: MongoDB
-   **Machine Learning**: Scikit-learn, Pandas, NumPy, Imbalanced-learn
-   **Deployment**: Streamlit, Cloudpickle, Joblib

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
