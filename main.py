import os
import shutil
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form,APIRouter, Query
from fastapi.responses import FileResponse
from pymongo import MongoClient
from dotenv import load_dotenv
from constants import *
from pydantic_models.user_model import UserRegister, UserLogin
from pydantic_models.dataset_upload import DatasetUploadResponse, TaskType , TrainDataset
from utils.jwt_handler import create_access_token, verify_token
from uuid import uuid4
import pandas as pd
from user_section.main import User
from io import BytesIO
from pathlib import Path
load_dotenv()

client = MongoClient(os.getenv("MONGO_DB_URI"))
db = client["MetaML"]
user_collection = db["users"]


app = FastAPI()


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


@app.get("/users/get_columns")
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

        user_folder = f"{USERS_FOLDER}/{user['username']}/datasets/{task_type.value}"
        os.makedirs(user_folder, exist_ok=True)

        unique_name = f"{uuid4().hex}_{file.filename}"
        target_path = f"{user_folder}/{unique_name}"

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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
    try : 
        start_train(train_dataset)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500,detail="An error occurred while training make sure that your target_col and task_type are correct, if still issue persists feel free to contact us!")
        

def start_train(
    train_dataset: TrainDataset
):
    user = User(
        train_dataset.dataset.path,
        train_dataset.dataset.user_id,
        train_dataset.target_col,
        train_dataset.tuning,
        train_dataset.dataset.task_type,
        train_dataset.dataset.stored_name.split(".")[0],
    )
    user.start()


@app.get("/users/get_models")
def get_models(user = Depends(verify_token)):
    user_base = Path(USERS_FOLDER) / user["username"] / "models"

    response = {
        "classification": [],
        "regression": []
    }

    for t in ["classification", "regression"]:
        folder = user_base / t

        if not folder.exists() or not folder.is_dir():
            continue

        for item in folder.iterdir():
            if not item.is_file():
                continue

            # Create relative path for download
            rel_path = item.relative_to(Path(USERS_FOLDER))

            response[t].append({
                "name": item.name,
                "path": str(rel_path),
                "download_url": f"/users/download_model?file_path={rel_path}",
            })

    return response


@app.get("/users/download_model")
def download_model(file_path: str = Query(...)):
    base_dir = Path(USERS_FOLDER).resolve()
    abs_path = (base_dir / file_path).resolve()

    # Security: prevent path traversal
    if base_dir not in abs_path.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=abs_path,
        filename=abs_path.name,
        media_type="application/octet-stream"
    )



@app.get("/users/get_datasets")
def get_all_datasets(user = Depends(verify_token)):
    base_path = f"{USERS_FOLDER}/{user['username']}/datasets"

    response = {"classification": [], "regression": []}

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
                "name": file.split("_",1)[1],
                "download_url": f"/users/download_dataset?file_path={full_path}"
            })

        response[t] = files

    return response

@app.get("/users/download_dataset")
def download_dataset(
    file_path: str,
    user = Depends(verify_token)
):
    # Only allow files inside user's directory
    user_base = f"{USERS_FOLDER}/{user['username']}/datasets"

    if not file_path.startswith(user_base):
        raise HTTPException(403, "Not allowed")

    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path).split("_",1)[1],
        media_type="application/octet-stream"
    )
