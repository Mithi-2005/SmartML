# SmartML

SmartML is an end-to-end AutoML platform that helps users upload tabular datasets, train machine learning models, compare results, and export ready-to-run prediction bundles. The project combines a FastAPI backend, a React frontend, and an ML pipeline that automates preprocessing, model selection, evaluation, and packaging.

## What the Project Does

SmartML is built for supervised learning on tabular CSV datasets.

- Supports both classification and regression tasks
- Lets users register, log in, and manage their own datasets, models, and exported bundles
- Automatically preprocesses uploaded datasets before training
- Uses a meta-learning stage to shortlist promising algorithms
- Trains and evaluates candidate models, with optional hyperparameter tuning
- Stores model metadata such as metrics, explanations, and generated timestamps
- Exports an inference bundle as a zip file with a Streamlit app for local prediction

## Core Workflow

1. A user uploads a CSV dataset and selects the target column and task type.
2. The backend saves the dataset under that user's storage area.
3. Training starts in a background thread so the API can return immediately.
4. The preprocessing pipeline cleans and transforms the data.
5. A meta-learning predictor narrows the model search space.
6. Candidate models are trained and evaluated.
7. The best model is saved along with metadata and explanations.
8. A downloadable bundle is created containing the trained model, serialized preprocessor, and a Streamlit prediction app.

## Main Features

### User and Project Management

- JWT-based authentication for register/login flows
- Per-user storage for datasets, status files, trained models, and exported bundles
- API endpoints to list, download, and delete user assets

### Data Handling

- CSV upload and automatic column preview
- Automatic train, validation, and test splitting
- Pipeline-driven preprocessing for tabular data

### Automated Preprocessing

The preprocessing system is designed for real-world tabular datasets and includes logic for:

- Missing value handling
- Categorical encoding
- Feature scaling
- High-correlation feature removal
- PCA when needed
- Inference-safe transformation through a serialized preprocessor artifact

### Model Training

SmartML trains different model families depending on the task.

Classification candidates include:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- SVC
- Random Forest
- Histogram Gradient Boosting

Regression candidates include:

- Linear Regression
- Polynomial Regression
- Ridge
- Lasso
- ElasticNet
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- SVR or LinearSVR
- Random Forest Regressor
- Histogram Gradient Boosting Regressor

Hyperparameter search is supported through:

- `GridSearchCV` for classification
- `RandomizedSearchCV` for regression

### Model Outputs

- Best-model persistence as `.pkl`
- Metadata stored as `.meta.json`
- Human-readable metric summaries
- Basic local explanations generated with LIME
- Training status tracking for frontend polling

### Bundle Export

Each exported bundle contains:

- `model.pkl`: trained estimator
- `preprocessor.pkl`: self-contained inference preprocessor
- `app.py`: Streamlit app for prediction
- `requirements.txt`: runtime dependencies for the bundle
- `README.md`: usage instructions for the exported artifact

The bundle is zipped and can be shared or run independently of the main SmartML app.

## Tech Stack

### Backend

- FastAPI
- Python
- MongoDB
- JWT authentication

### Machine Learning

- scikit-learn
- pandas
- numpy
- imbalanced-learn
- xgboost
- LIME
- cloudpickle

### Frontend

- React
- Vite
- React Router
- Framer Motion

## Project Structure

```text
SmartML/
|-- main.py                              # FastAPI entry point and API routes
|-- config.py                            # JWT-related configuration
|-- constants/                           # Shared path constants
|-- components/                          # Core preprocessing and training utilities
|   |-- preprocessing.py
|   |-- training.py
|   `-- meta_features_extraction.py
|-- user_section/                        # User-facing training, prediction, and status logic
|   |-- main.py
|   |-- prediction/
|   |-- training/
|   `-- pending_datasets/
|-- meta_model/                          # Meta-model training assets
|-- pydantic_models/                     # Request and response schemas
|-- utils/                               # JWT and DB helpers
|-- storage/                             # Generated user files
|-- Frontend/automl/                     # React frontend
|-- requirements.txt
`-- README.md
```

## Backend API Overview

Important endpoints in the current backend:

- `POST /users/register`
- `POST /users/login`
- `GET /users/profile`
- `POST /users/get_columns`
- `POST /users/send_dataset`
- `GET /users/training_status`
- `GET /users/active_training_runs`
- `GET /users/get_models`
- `GET /users/get_bundles`
- `GET /users/get_datasets`
- `GET /users/download_model`
- `GET /users/download_bundle`
- `GET /users/download_dataset`
- `DELETE /users/delete_model`
- `DELETE /users/delete_bundle`
- `DELETE /users/delete_dataset`

Interactive API docs are available through FastAPI Swagger once the backend is running.

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+ recommended for the frontend
- MongoDB instance accessible from the backend

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd SmartML
```

### 2. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
MONGO_DB_URI=mongodb://localhost:27017/
JWT_SECRET=your_secret_key_here
```

Notes:

- `main.py` expects `MONGO_DB_URI`
- JWT settings are read from `config.py`, where `JWT_SECRET` can be overridden through the environment

## Running the Application

### Start the Backend

```bash
uvicorn main:app --reload
```

The backend will be available at:

- API base: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

### Start the Frontend

From `Frontend/automl`:

```bash
npm install
npm run dev
```

By default, the frontend talks to `http://localhost:8000`.

If needed, create `Frontend/automl/.env` and set:

```env
VITE_API_BASE=http://localhost:8000
```

The frontend dev server typically runs on `http://localhost:5173`.

## Storage Layout

Generated user assets are stored under `storage/<username>/` and are organized into folders such as:

- `datasets/`
- `models/`
- `templates/`
- `status/`

This makes it easier to manage uploaded files, trained models, downloadable bundles, and background training progress separately for each user.

## Exported Bundle Usage

After downloading a bundle:

1. Extract the zip file.
2. Install the included dependencies.
3. Launch the bundled Streamlit app.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The bundle expects a CSV with the same feature schema used during training. Predictions are returned in a table and can be downloaded as a CSV.

## Current Scope and Notes

- The project is focused on tabular CSV datasets
- Training is asynchronous and tracked through status JSON files
- The frontend and backend are separate apps that run independently in development
- Exported bundles are designed for inference, not retraining

## Future Improvements

Some natural next steps for the project could include:

- Better deployment options beyond local Streamlit bundles
- Experiment tracking and run history dashboards
- More detailed model comparison views
- Dataset validation and richer upload feedback
- Docker-based local setup

## Author

Developed by K V Mithilesh  
GitHub: [@Mithi2005](https://github.com/Mithi2005)
