from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.data_service import load_dataset_from_upload
from services.ml_service import get_algorithm_catalog, run_algorithm


router = APIRouter(prefix="/api/weka", tags=["weka"])


@router.get("/algorithms")
async def get_algorithms():
    return get_algorithm_catalog()


@router.post("/classify")
async def classify(
    file: UploadFile = File(...),
    algorithm: str = Form(...),
    evaluationMethod: str = Form("crossvalidation"),
    folds: int = Form(10),
    trainPercent: float = Form(66),
    seed: int = Form(1),
):
    try:
        content = await file.read()
        dataset = load_dataset_from_upload(file, content)
        return run_algorithm(
            dataset=dataset,
            algorithm_id=algorithm,
            evaluation_method=evaluationMethod,
            folds=folds,
            train_percent=trainPercent,
            seed=seed,
            dataset_name=file.filename or "dataset",
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Unexpected error: {error}") from error
