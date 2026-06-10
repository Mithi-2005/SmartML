import os
import shutil
import json
import bcrypt
import threading
import logging
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form,APIRouter, Query
from fastapi.responses import FileResponse
from pymongo import MongoClient
from dotenv import load_dotenv
from constants import *
from pydantic_models.user_model import UserRegister, UserLogin
from pydantic_models.dataset_upload import DatasetUploadResponse, TaskType , TrainDataset
from utils.jwt_handler import create_access_token, verify_token
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import pandas as pd
from user_section.main import User
from user_section.training.status_tracker import TrainingStatusTracker
from utils.azure_blob import azure_blob_helper
from io import BytesIO
from pathlib import Path
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

client = MongoClient(os.getenv("MONGO_DB_URI"))
db = client["MetaML"]
user_collection = db["users"]


def download_meta_models():
    """Downloads any missing gitignored meta-models from Azure Blob Storage on startup."""
    if not azure_blob_helper.enabled:
        logging.info("[AZURE] Azure connection string not configured. Skipping startup meta-model download.")
        return
    
    meta_files = [
        ("meta_model/models/meta_classification_model.pkl", META_CLASSIFICATION_MODEL),
        ("meta_model/models/meta_regression_model.pkl", META_REGRESSION_MODEL),
        ("meta_model/datasets/meta_features_classification.csv", META_CLASSIFICATION_DATASET),
        ("meta_model/datasets/meta_features_regression.csv", META_REGRESSION_DATASET),
        ("meta_model/results/meta_dataset_results.csv", META_RESULTS_PATH),
    ]
    
    for blob_name, local_path_str in meta_files:
        local_path = Path(local_path_str)
        if not local_path.exists():
            logging.info(f"[AZURE] Downloading gitignored meta-model resource {blob_name} to {local_path}...")
            azure_blob_helper.download_file(blob_name, local_path)

# Run download check on import
download_meta_models()


app = FastAPI()


CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:8000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def _normalize_artifact_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_", 1)
    if len(parts) == 2 and len(parts[0]) == 32:
        stem = parts[1]
    return stem.replace("_", " ").replace("-", " ").strip().lower()


def _load_metadata_map(base: Path, task: str):
    task_dir = base / task
    metadata_map = {}

    if not task_dir.exists() or not task_dir.is_dir():
        return metadata_map

    for meta_file in task_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            normalized_name = _normalize_artifact_name(meta_file.name.replace(".meta.json", ""))
            metadata_map[normalized_name] = meta
        except Exception as meta_error:
            logging.warning(f"[BUNDLES] Failed to index metadata {meta_file}: {meta_error}")

    return metadata_map


def _load_model_metadata_map(models_base: Path, task: str):
    return _load_metadata_map(models_base, task)



@app.get("/")
def root():
    return {"msg": "Welcome to Metaml"}


@app.post("/users/register")
def register(user: UserRegister):

    print("PASSWORD LENGTH:", len(user.password))

    if not all(
        [
            user.fname,
            user.lname,
            user.username,
            user.email,
            user.password,
            user.cpassword,
        ]
    ):
        raise HTTPException(status_code=400, detail="all fields are required")

    if user_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="username already taken")

    if user_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already exists!")

    if user.password != user.cpassword:
        raise HTTPException(status_code=400, detail="password doesn't match")

    hashed_pass = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt()).decode(
        "utf-8"
    )

    user_data = {
        "fname": user.fname,
        "lname": user.lname,
        "username": user.username,
        "email": user.email,
        "password": hashed_pass,
    }

    user_collection.insert_one(user_data)

    return {"msg": "User Added Successfully"}


@app.post("/users/login", status_code=200)
def login(user: UserLogin):
    if not all([user.email, user.password]):
        raise HTTPException(status_code=400, detail="All fields are required")

    existing = db["users"].find_one({"email": user.email})

    if not existing:
        raise HTTPException(status_code=404, detail="Email not found")

    if not bcrypt.checkpw(
        user.password.encode("utf-8"), existing["password"].encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"email": user.email})

    return {"message": "Login Successful", "token": token}


@app.get("/users/profile")
def profile(user=Depends(verify_token)):
    return {"user": user}


@app.post("/users/get_columns")
def get_columns(file : UploadFile = File(...),user=Depends(verify_token)):
    
    content=file.file.read()
    df = pd.read_csv(BytesIO(content))  
    return list(df.columns)



@app.post("/users/send_dataset")
def get_dataset(
    task_type: TaskType = Form(...),
    user=Depends(verify_token),
    file: UploadFile = File(...),
    target_col : str = Form(...),
    tuning : bool = Form(...)
):
    
    try : 
        print(task_type, file)
        if not file.filename:
            raise HTTPException(status_code=400, detail="file not found")

        unique_name = f"{uuid4().hex}_{file.filename}"
        dataset_id = Path(unique_name).stem

        if azure_blob_helper.enabled:
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "smartml_datasets"
            temp_dir.mkdir(parents=True, exist_ok=True)
            target_path = str(temp_dir / unique_name)
        else:
            user_folder = f"{USERS_FOLDER}/{user['username']}/datasets/{task_type.value}"
            os.makedirs(user_folder, exist_ok=True)
            target_path = f"{user_folder}/{unique_name}"

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload to Azure Blob Storage if enabled
        if azure_blob_helper.enabled:
            try:
                blob_name = f"storage/{user['username']}/datasets/{task_type.value}/{unique_name}"
                azure_blob_helper.upload_file(Path(target_path), blob_name)
            except Exception as azure_err:
                logging.error(f"[AZURE] Failed to upload dataset {target_path} to Azure: {azure_err}")


        dataset = DatasetUploadResponse(
            original_name=file.filename,
            stored_name=unique_name,
            user_id=user["username"],
            path=str(target_path),
            task_type=task_type.value,
        )
        
        
        train_dataset=TrainDataset(
            dataset=dataset,
            tuning=tuning,
            target_col=target_col
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500,detail="Internal Server Error! Dont worry it is not your fault!")
    
    # Start training in background thread
    dataset_id = Path(unique_name).stem
    thread = threading.Thread(
        target=start_train_async,
        args=(train_dataset,),
        daemon=True
    )
    thread.start()
    logging.info(f"[UPLOAD] Started background training for dataset {dataset_id}")

    # Return immediately with dataset_id for polling
    return {
        "message": "Dataset uploaded successfully. Training started.",
        "dataset": {
            "original_name": file.filename,
            "stored_name": unique_name,
            "task_type": task_type.value,
            "target_col": target_col,
            "dataset_id": dataset_id,
        },
        "status": {
            "dataset_id": dataset_id,
            "poll_url": f"/users/training_status?dataset_id={dataset_id}",
        },
    }

def start_train_async(train_dataset: TrainDataset):
    """Run training in background thread with error handling"""
    dataset_id = Path(train_dataset.dataset.stored_name).stem
    try:
        logging.info(f"[TRAINING] Starting async training for dataset {dataset_id}")
        start_train(train_dataset)
        logging.info(f"[TRAINING] Completed training for dataset {dataset_id}")
    except Exception as e:
        logging.error(f"[TRAINING] Failed for dataset {dataset_id}: {str(e)}", exc_info=True)
        try:
            tracker = TrainingStatusTracker(train_dataset.dataset.user_id, dataset_id)
            tracker.error(f"Training failed: {str(e)}")
        except Exception as tracker_error:
            logging.error(f"[TRAINING] Failed to update status tracker: {tracker_error}")

def start_train(
    train_dataset: TrainDataset
):
    dataset_id = Path(train_dataset.dataset.stored_name).stem
    tracker = TrainingStatusTracker(train_dataset.dataset.user_id, dataset_id)
    tracker.update("queued", "Dataset received. Preparing preprocessing pipeline.")
    user = User(
        train_dataset.dataset.path,
        train_dataset.dataset.user_id,
        train_dataset.target_col,
        train_dataset.tuning,
        train_dataset.dataset.task_type,
        dataset_id,
        status_tracker=tracker,
    )
    return user.start()


from fastapi.responses import StreamingResponse

def _load_azure_metadata_map(username: str, task: str, category: str):
    """
    Loads all .meta.json files for a user from Azure Blob Storage directly in memory.
    category: "models" or "templates"
    """
    metadata_map = {}
    if not azure_blob_helper.enabled:
        return metadata_map

    prefix = f"storage/{username}/{category}/{task}/"
    try:
        blobs = azure_blob_helper.list_files(prefix)
        for blob_name in blobs:
            if blob_name.endswith(".meta.json"):
                meta_content = azure_blob_helper.download_blob_to_memory(blob_name)
                if meta_content:
                    try:
                        meta = json.loads(meta_content)
                        filename = blob_name.split("/")[-1]
                        normalized_name = _normalize_artifact_name(filename.replace(".meta.json", ""))
                        metadata_map[normalized_name] = meta
                    except Exception as meta_error:
                        logging.warning(f"[BUNDLES] Failed to parse Azure metadata {blob_name}: {meta_error}")
    except Exception as e:
        logging.error(f"[AZURE] Failed to load metadata map for {prefix}: {e}")

    return metadata_map


@app.get("/users/get_models")
def get_models(user = Depends(verify_token)):
    response = {
        "classification": [],
        "regression": []
    }

    if azure_blob_helper.enabled:
        username = user["username"]
        for t in ["classification", "regression"]:
            meta_map = _load_azure_metadata_map(username, t, "models")
            
            prefix = f"storage/{username}/models/{t}/"
            blobs = azure_blob_helper.list_files(prefix)
            for blob_name in blobs:
                if blob_name.endswith(".pkl"):
                    try:
                        rel_path = str(Path(blob_name).relative_to("storage")).replace("\\", "/")
                    except Exception:
                        rel_path = blob_name
                    
                    filename = blob_name.split("/")[-1]
                    normalized_name = _normalize_artifact_name(filename.replace(".pkl", ""))
                    meta = meta_map.get(normalized_name, {})
                    
                    response[t].append({
                        "name": filename,
                        "path": rel_path,
                        "download_url": f"/users/download_model?file_path={rel_path}",
                        "metric_name": meta.get("metric_name"),
                        "metric_value": meta.get("metric_value"),
                        "explanations": meta.get("explanations", []),
                        "generated_at": meta.get("generated_at"),
                        "model_label": meta.get("model_name"),
                        "model_reason": meta.get("model_reason"),
                        "human_metric": meta.get("human_metric"),
                    })
        return response
    else:
        user_base = Path(USERS_FOLDER) / user["username"] / "models"
        for t in ["classification", "regression"]:
            folder = user_base / t
            if not folder.exists() or not folder.is_dir():
                continue
            for item in folder.iterdir():
                if not item.is_file() or item.suffix != ".pkl":
                    continue
                rel_path = item.relative_to(Path(USERS_FOLDER))
                meta = {}
                meta_file = item.with_suffix(".meta.json")
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    except Exception as meta_error:
                        print(f"[META] Failed to read {meta_file}: {meta_error}")
                response[t].append({
                    "name": item.name,
                    "path": str(rel_path),
                    "download_url": f"/users/download_model?file_path={rel_path}",
                    "metric_name": meta.get("metric_name"),
                    "metric_value": meta.get("metric_value"),
                    "explanations": meta.get("explanations", []),
                    "generated_at": meta.get("generated_at"),
                    "model_label": meta.get("model_name"),
                    "model_reason": meta.get("model_reason"),
                    "human_metric": meta.get("human_metric"),
                })
        return response


@app.get("/users/download_model")
def download_model(file_path: str = Query(...)):
    if azure_blob_helper.enabled:
        blob_name = f"storage/{file_path}" if not file_path.startswith("storage/") else file_path
        filename = blob_name.split("/")[-1]
        try:
            stream = azure_blob_helper.get_blob_stream(blob_name)
            return StreamingResponse(stream, media_type="application/octet-stream", headers={
                "Content-Disposition": f"attachment; filename={filename}"
            })
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Blob not found: {e}")
            
    base_dir = Path(USERS_FOLDER).resolve()
    abs_path = (base_dir / file_path).resolve()
    if base_dir not in abs_path.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=abs_path,
        filename=abs_path.name,
        media_type="application/octet-stream"
    )


@app.get("/users/get_bundles")
def get_bundles(user=Depends(verify_token)):
    response = {"classification": [], "regression": []}

    if azure_blob_helper.enabled:
        username = user["username"]
        for task in response.keys():
            models_meta = _load_azure_metadata_map(username, task, "models")
            templates_meta = _load_azure_metadata_map(username, task, "templates")
            models_meta.update(templates_meta)
            
            prefix = f"storage/{username}/templates/{task}/"
            try:
                blobs = azure_blob_helper.container_client.list_blobs(name_starts_with=prefix)
                for blob in blobs:
                    if blob.name.endswith(".zip"):
                        try:
                            rel_path = str(Path(blob.name).relative_to("storage")).replace("\\", "/")
                        except Exception:
                            rel_path = blob.name
                            
                        filename = blob.name.split("/")[-1]
                        normalized_name = _normalize_artifact_name(filename.replace(".zip", ""))
                        meta = models_meta.get(normalized_name, {})
                        
                        response[task].append({
                            "name": filename.replace(".zip", ""),
                            "display_name": _normalize_artifact_name(filename.replace(".zip", "")).title(),
                            "path": rel_path,
                            "size_bytes": blob.size,
                            "modified_ts": blob.last_modified.timestamp() if blob.last_modified else 0,
                            "download_url": f"/users/download_bundle?file_path={rel_path}",
                            "model_name": meta.get("model_name"),
                            "metric_name": meta.get("metric_name"),
                            "metric_value": meta.get("metric_value"),
                            "explanations": meta.get("explanations", []),
                            "model_reason": meta.get("model_reason"),
                            "human_metric": meta.get("human_metric"),
                            "generated_at": meta.get("generated_at"),
                        })
            except Exception as e:
                logging.error(f"[AZURE] Failed to list bundles: {e}")
        return response
    else:
        templates_base = Path(USERS_FOLDER) / user["username"] / "templates"
        models_base = Path(USERS_FOLDER) / user["username"] / "models"
        for task in response.keys():
            folder = templates_base / task
            if not folder.exists():
                continue
            metadata_map = _load_model_metadata_map(models_base, task)
            metadata_map.update(_load_metadata_map(templates_base, task))

            for zip_file in folder.glob("*.zip"):
                rel_path = zip_file.relative_to(Path(USERS_FOLDER))
                stats = zip_file.stat()
                meta = metadata_map.get(_normalize_artifact_name(zip_file.stem), {})

                response[task].append(
                    {
                        "name": zip_file.stem,
                        "display_name": _normalize_artifact_name(zip_file.stem).title(),
                        "path": str(rel_path),
                        "size_bytes": stats.st_size,
                        "modified_ts": stats.st_mtime,
                        "download_url": f"/users/download_bundle?file_path={rel_path}",
                        "model_name": meta.get("model_name"),
                        "metric_name": meta.get("metric_name"),
                        "metric_value": meta.get("metric_value"),
                        "explanations": meta.get("explanations", []),
                        "model_reason": meta.get("model_reason"),
                        "human_metric": meta.get("human_metric"),
                        "generated_at": meta.get("generated_at"),
                    }
                )
        return response


@app.get("/users/download_bundle")
def download_bundle(file_path: str = Query(...), user=Depends(verify_token)):
    if azure_blob_helper.enabled:
        blob_name = f"storage/{file_path}" if not file_path.startswith("storage/") else file_path
        filename = blob_name.split("/")[-1]
        try:
            stream = azure_blob_helper.get_blob_stream(blob_name)
            return StreamingResponse(stream, media_type="application/zip", headers={
                "Content-Disposition": f"attachment; filename={filename}"
            })
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Bundle not found in Azure: {e}")
            
    base_dir = (Path(USERS_FOLDER) / user["username"] / "templates").resolve()
    abs_path = (Path(USERS_FOLDER) / file_path).resolve()
    if base_dir not in abs_path.parents:
        raise HTTPException(status_code=403, detail="Not allowed")
    if not abs_path.exists() or abs_path.suffix.lower() != ".zip":
        raise HTTPException(status_code=404, detail="Bundle not found")
    return FileResponse(
        path=abs_path,
        filename=abs_path.name,
        media_type="application/zip"
    )


@app.delete("/users/delete_model")
def delete_model(file_path: str = Query(...), user=Depends(verify_token)):
    if azure_blob_helper.enabled:
        azure_blob_helper.delete_file(f"storage/{file_path}")
        azure_blob_helper.delete_file(f"storage/{file_path}".replace(".pkl", ".meta.json"))

    user_base = (Path(USERS_FOLDER) / user["username"] / "models").resolve()
    abs_path = (Path(USERS_FOLDER) / file_path).resolve()
    if user_base in abs_path.parents and abs_path.exists():
        abs_path.unlink()
        meta_file = abs_path.with_suffix(".meta.json")
        if meta_file.exists():
            meta_file.unlink()

    return {"msg": "Model deleted"}


@app.delete("/users/delete_bundle")
def delete_bundle(file_path: str = Query(...), user=Depends(verify_token)):
    if azure_blob_helper.enabled:
        azure_blob_helper.delete_file(f"storage/{file_path}")
        azure_blob_helper.delete_file(f"storage/{file_path}".replace(".zip", ".meta.json"))

    user_base = (Path(USERS_FOLDER) / user["username"] / "templates").resolve()
    abs_path = (Path(USERS_FOLDER) / file_path).resolve()
    if user_base in abs_path.parents and abs_path.exists():
        abs_path.unlink()
        meta_path = abs_path.with_suffix(".meta.json")
        if meta_path.exists():
            meta_path.unlink()

    logging.info(f"[DELETE] Bundle deleted: {file_path}")
    return {"msg": "Bundle deleted"}


@app.get("/users/get_datasets")
def get_all_datasets(user = Depends(verify_token)):
    response = {"classification": [], "regression": []}

    if azure_blob_helper.enabled:
        username = user["username"]
        for t in ["classification", "regression"]:
            prefix = f"storage/{username}/datasets/{t}/"
            try:
                blobs = azure_blob_helper.list_files(prefix)
                files = []
                for blob_name in blobs:
                    filename = blob_name.split("/")[-1]
                    files.append({
                        "name": filename.split("_", 1)[1] if "_" in filename else filename,
                        "download_url": f"/users/download_dataset?file_path={blob_name}",
                        "path": blob_name
                    })
                response[t] = files
            except Exception as e:
                logging.error(f"[AZURE] Failed to list datasets: {e}")
        return response
    else:
        base_path = f"{USERS_FOLDER}/{user['username']}/datasets"
        for t in ["classification", "regression"]:
            folder = f"{base_path}/{t}"
            if not os.path.exists(folder):
                response[t] = []
                continue
            files = []
            for file in os.listdir(folder):
                full_path = os.path.join(folder, file)
                if not os.path.isfile(full_path):
                    continue
                files.append({
                    "name": file.split("_",1)[1] if "_" in file else file,
                    "download_url": f"/users/download_dataset?file_path={full_path}",
                    "path": full_path
                })
            response[t] = files
        return response


@app.get("/users/download_dataset")
def download_dataset(file_path: str, user = Depends(verify_token)):
    if azure_blob_helper.enabled:
        blob_name = f"storage/{file_path}" if not file_path.startswith("storage/") else file_path
        if not blob_name.startswith(f"storage/{user['username']}/datasets/"):
            raise HTTPException(403, "Not allowed")
        filename = blob_name.split("/")[-1]
        display_filename = filename.split("_", 1)[1] if "_" in filename else filename
        try:
            stream = azure_blob_helper.get_blob_stream(blob_name)
            return StreamingResponse(stream, media_type="application/octet-stream", headers={
                "Content-Disposition": f"attachment; filename={display_filename}"
            })
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Dataset not found in Azure: {e}")
            
    user_base = f"{USERS_FOLDER}/{user['username']}/datasets"
    if not file_path.startswith(user_base):
        raise HTTPException(403, "Not allowed")
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path).split("_",1)[1] if "_" in os.path.basename(file_path) else os.path.basename(file_path),
        media_type="application/octet-stream"
    )


@app.delete("/users/delete_dataset")
def delete_dataset_entry(file_path: str = Query(...), user=Depends(verify_token)):
    if azure_blob_helper.enabled:
        blob_name = f"storage/{file_path}" if not file_path.startswith("storage/") else file_path
        if blob_name.startswith(f"storage/{user['username']}/datasets/"):
            azure_blob_helper.delete_file(blob_name)

    user_base = Path(USERS_FOLDER) / user["username"] / "datasets"
    abs_path = Path(file_path).resolve()
    if user_base.resolve() in abs_path.parents and abs_path.exists():
        abs_path.unlink()

    return {"msg": "Dataset deleted"}


@app.get("/users/training_status")
def get_training_status(dataset_id: str = Query(...), user=Depends(verify_token)):
    run_id = f"{user['username']}_{dataset_id}"
    try:
        status_doc = db["training_status"].find_one({"_id": run_id})
        if not status_doc:
            return {
                "dataset": dataset_id,
                "history": [],
                "current": None,
            }
        status_doc.pop("_id", None)
        return status_doc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read training status: {exc}")


@app.get("/users/active_training_runs")
def get_active_training_runs(user=Depends(verify_token)):
    """Get all active (non-terminal) training runs for the authenticated user"""
    active_runs = []
    try:
        cursor = db["training_status"].find({
            "user_id": user["username"],
            "current.state": {"$nin": ["completed", "error"]}
        })
        for run in cursor:
            current = run.get("current", {})
            dataset_id = run.get("dataset")
            active_runs.append({
                "dataset_id": dataset_id,
                "name": dataset_id,
                "current_phase": current.get("phase"),
                "current_state": current.get("state"),
            })
    except Exception as e:
        logging.warning(f"Failed to query active training runs: {e}")

    return {"active_runs": active_runs}

