import base64
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supply_iq_core import SupplyIQ

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

app = FastAPI(title="SupplyIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = SupplyIQ(api_key=API_KEY)
TEMP_DIR = Path(tempfile.gettempdir()) / "supplyiq_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class AskPayload(BaseModel):
    query: str


class ScenarioPayload(BaseModel):
    scenario_name: str
    custom_percent: Optional[float] = 30
    custom_scenario_text: Optional[str] = ""


@app.get("/health")
def health():
    return {"status": "ok", "app": "SupplyIQ API"}


@app.get("/dashboard")
def dashboard():
    return {"dashboard_html": system.get_kpi_dashboard()}


@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or "data.csv").suffix or ".csv"
        temp_path = TEMP_DIR / f"erp_upload{suffix}"

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        status = system.load_erp_data(str(temp_path))

        return {
            "status": status,
            "dashboard_html": system.get_kpi_dashboard(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV upload failed: {str(e)}",
        )


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    suffix = Path(file.filename or "contract.pdf").suffix or ".pdf"
    temp_path = TEMP_DIR / f"contract_upload{suffix}"
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    status = system.setup_document_intelligence(pdf_file_path=str(temp_path))
    return {"status": status, "dashboard_html": system.get_kpi_dashboard()}


@app.post("/forecast")
def forecast():
    fig, summary, records, metrics = system.run_forecast(periods=30)
    if fig is None:
        raise HTTPException(status_code=400, detail=summary)

    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", bbox_inches="tight", dpi=150)
    img_bytes.seek(0)
    plot_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

    return {
        "summary": summary,
        "records": records,
        "metrics": metrics,
        "plot_base64": plot_base64,
        "dashboard_html": system.get_kpi_dashboard(),
    }


@app.post("/anomalies")
def anomalies():
    summary, records, metrics = system.detect_anomalies()
    if summary.startswith("Error:"):
        raise HTTPException(status_code=400, detail=summary)

    return {
        "summary": summary,
        "records": records,
        "metrics": metrics,
        "dashboard_html": system.get_kpi_dashboard(),
    }


@app.post("/contract-report")
def contract_report():
    report = system.generate_contract_report()
    if report.startswith("Knowledge base not initialized") or report.startswith("No relevant"):
        raise HTTPException(status_code=400, detail=report)

    return {"report": report, "dashboard_html": system.get_kpi_dashboard()}


@app.post("/ask-contract")
def ask_contract(payload: AskPayload):
    answer = system.query_documents(payload.query)
    if answer.startswith("Knowledge base not initialized") or answer.startswith("Please enter"):
        raise HTTPException(status_code=400, detail=answer)

    return {"answer": answer, "dashboard_html": system.get_kpi_dashboard()}


@app.post("/decision")
def decision():
    report = system.generate_decision()
    if report.startswith("Error:"):
        raise HTTPException(status_code=400, detail=report)

    return {"report": report, "dashboard_html": system.get_kpi_dashboard()}


@app.post("/scenario")
def scenario(payload: ScenarioPayload):
    scenario_name = payload.scenario_name
    custom_text = (payload.custom_scenario_text or "").strip()

    if custom_text:
        scenario_name = custom_text
    elif scenario_name == "Demand increase (custom %)":
        custom_percent = payload.custom_percent if payload.custom_percent is not None else 30
        scenario_name = f"Demand increases by {custom_percent:g}%"

    result = system.run_scenario(scenario_name)
    if result.startswith("Error:"):
        raise HTTPException(status_code=400, detail=result)

    return {"report": result, "dashboard_html": system.get_kpi_dashboard()}
