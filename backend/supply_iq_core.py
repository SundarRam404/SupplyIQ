import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from groq import Groq


class SupplyIQ:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama-3.3-70b-versatile"

        self.df: Optional[pd.DataFrame] = None

        # Document intelligence state
        self.doc_chunks: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None
        self.doc_filename: Optional[str] = None

        # Cached outputs
        self.last_forecast_summary: Optional[str] = None
        self.last_forecast_records: Optional[List[Dict]] = None
        self.last_forecast_metrics: Optional[Dict] = None

        self.last_anomaly_summary: Optional[str] = None
        self.last_anomaly_records: Optional[List[Dict]] = None
        self.last_anomaly_metrics: Optional[Dict] = None

        self.last_contract_report: Optional[str] = None
        self.last_contract_risk: str = "UNKNOWN"

        self.last_decision_report: Optional[str] = None
        self.last_ai_risk_snapshot: Optional[Dict] = None

    # =========================================================
    # Helpers
    # =========================================================
    def _safe_float(self, value, default=0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def _risk_badge(self, value: str) -> str:
        value = (value or "UNKNOWN").upper()
        if value == "LOW":
            return "🟢 LOW"
        if value == "MEDIUM":
            return "🟠 MEDIUM"
        if value == "HIGH":
            return "🔴 HIGH"
        if value == "NOT RUN":
            return "⚪ NOT RUN"
        return "⚪ UNKNOWN"

    def _chunk_text(self, text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        text = text.strip()

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - overlap

        return chunks

    def _retrieve_relevant_context(self, query: str, top_k: int = 4) -> str:
        if self.vectorizer is None or self.doc_matrix is None or not self.doc_chunks:
            return ""

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        if len(scores) == 0:
            return ""

        top_indices = scores.argsort()[::-1][:top_k]
        selected = [self.doc_chunks[i] for i in top_indices if scores[i] > 0]

        return "\n\n".join(selected)

    # (removed _estimate_inventory_risk)

    def _build_kpi_html(self, demand_risk: str, contract_risk: str, inventory_risk: str) -> str:
        def card(title: str, value: str) -> str:
            color = {
                "LOW": "#22c55e",
                "MEDIUM": "#f59e0b",
                "HIGH": "#ef4444",
                "NOT RUN": "#94a3b8",
                "UNKNOWN": "#94a3b8",
            }.get(value.upper(), "#94a3b8")
            return f"""
            <div style="
                flex:1;
                min-width:220px;
                background: linear-gradient(135deg, #101828 0%, #1e293b 100%);
                border: 1px solid #334155;
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.22);
            ">
                <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 8px;">{title}</div>
                <div style="font-size: 28px; font-weight: 800; color: {color};">{value.upper()}</div>
            </div>
            """

        return f"""
        <div style="display:flex; gap:16px; flex-wrap:wrap; margin: 10px 0 4px 0;">
            {card("Demand Forecast Risk", demand_risk)}
            {card("Supplier Contract Risk", contract_risk)}
            {card("Inventory Risk", inventory_risk)}
        </div>
        """
    def _llm_generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SupplyIQ, an elite supply chain intelligence engine. "
                        "Be sharp, executive-level, structured, and practical."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _build_data_profile(self) -> str:
        if self.df is None or self.df.empty:
            return "No ERP data uploaded."

        avg_demand = float(self.df["y"].mean())
        max_demand = float(self.df["y"].max())
        min_demand = float(self.df["y"].min())
        demand_std = float(self.df["y"].std()) if len(self.df) > 1 else 0.0
        avg_delay = float(self.df["delivery_delay_days"].mean()) if "delivery_delay_days" in self.df.columns else 0.0
        avg_price = float(self.df["component_price"].mean()) if "component_price" in self.df.columns else 0.0

        return f"""
ERP DATA PROFILE
- Rows: {len(self.df)}
- Date range: {self.df['ds'].min()} to {self.df['ds'].max()}
- Average demand: {avg_demand:.2f}
- Peak demand: {max_demand:.2f}
- Minimum demand: {min_demand:.2f}
- Demand std dev: {demand_std:.2f}
- Average delivery delay days: {avg_delay:.2f}
- Average component price: {avg_price:.2f}
""".strip()

    def _build_contract_profile(self) -> str:
        if not self.doc_chunks:
            return "No supplier contract uploaded."

        context = self._retrieve_relevant_context(
            "payment terms penalties delay obligations liability exclusivity lead time termination pricing risk"
        )

        if not context:
            context = "\n\n".join(self.doc_chunks[:2])

        return f"""
CONTRACT PROFILE
- Indexed chunks: {len(self.doc_chunks)}
- Document: {self.doc_filename or 'Unknown'}
- Relevant context:
{context[:3000]}
""".strip()

    def _get_ai_risk_snapshot(self) -> Dict:
        if self.last_ai_risk_snapshot is not None:
            return self.last_ai_risk_snapshot

        has_erp = self.df is not None and not self.df.empty
        has_contract = bool(self.doc_chunks)

        if not has_erp and not has_contract:
            snapshot = {
                "demand_risk": "NOT RUN",
                "contract_risk": "NOT RUN",
                "inventory_risk": "NOT RUN",
                "demand_reason": "No ERP data uploaded yet.",
                "contract_reason": "No supplier contract uploaded yet.",
                "inventory_reason": "No ERP data uploaded yet.",
                "demand_evidence": "Waiting for CSV upload.",
                "contract_evidence": "Waiting for PDF upload.",
                "inventory_evidence": "Waiting for CSV upload.",
                "raw_response": "No inputs uploaded.",
            }
            self.last_ai_risk_snapshot = snapshot
            return snapshot

        if not has_erp:
            snapshot = {
                "demand_risk": "NOT RUN",
                "contract_risk": "NOT RUN" if not has_contract else "MEDIUM",
                "inventory_risk": "NOT RUN",
                "demand_reason": "No ERP data uploaded yet.",
                "contract_reason": "Contract uploaded, but full contract risk analysis has limited operational evidence.",
                "inventory_reason": "No ERP data uploaded yet.",
                "demand_evidence": "Waiting for CSV upload.",
                "contract_evidence": f"Indexed chunks: {len(self.doc_chunks)}",
                "inventory_evidence": "Waiting for CSV upload.",
                "raw_response": "ERP data missing.",
            }
            self.last_ai_risk_snapshot = snapshot
            return snapshot

        if not has_contract:
            data_profile = self._build_data_profile()
            forecast_metrics = self.last_forecast_metrics or {}
            anomaly_metrics = self.last_anomaly_metrics or {}

            prompt = f"""
You are SupplyIQ's Risk Scoring AI.

Calculate ONLY these two risks from ERP evidence:
1. DEMAND_FORECAST_RISK
2. INVENTORY_RISK

IMPORTANT:
- Do not invent contract risk.
- Contract risk must be NOT RUN because no supplier contract has been uploaded.
- Base your reasoning only on ERP data, forecast metrics, and anomaly metrics.

Return EXACTLY in this format:

DEMAND_FORECAST_RISK: LOW or MEDIUM or HIGH
INVENTORY_RISK: LOW or MEDIUM or HIGH
DEMAND_REASON: <one sentence>
INVENTORY_REASON: <one sentence>
DEMAND_EVIDENCE: <short evidence line>
INVENTORY_EVIDENCE: <short evidence line>

ERP / DEMAND CONTEXT:
{data_profile}

FORECAST METRICS:
{forecast_metrics}

ANOMALY METRICS:
{anomaly_metrics}
""".strip()

            result = self._llm_generate(prompt)

            def extract(label: str, default: str = "MEDIUM") -> str:
                match = re.search(rf"{label}:\s*(LOW|MEDIUM|HIGH)", result, re.IGNORECASE)
                return match.group(1).upper() if match else default

            def extract_text(label: str, default: str) -> str:
                match = re.search(rf"{label}:\s*(.+)", result)
                return match.group(1).strip() if match else default

            snapshot = {
                "demand_risk": extract("DEMAND_FORECAST_RISK"),
                "contract_risk": "NOT RUN",
                "inventory_risk": extract("INVENTORY_RISK"),
                "demand_reason": extract_text("DEMAND_REASON", "AI demand risk reasoning unavailable."),
                "contract_reason": "No supplier contract uploaded yet.",
                "inventory_reason": extract_text("INVENTORY_REASON", "AI inventory risk reasoning unavailable."),
                "demand_evidence": extract_text("DEMAND_EVIDENCE", "ERP demand profile + forecast/anomaly metrics"),
                "contract_evidence": "Waiting for PDF upload.",
                "inventory_evidence": extract_text("INVENTORY_EVIDENCE", "ERP demand profile + forecast/anomaly metrics"),
                "raw_response": result,
            }
            self.last_ai_risk_snapshot = snapshot
            return snapshot

        data_profile = self._build_data_profile()
        contract_profile = self._build_contract_profile()
        forecast_metrics = self.last_forecast_metrics or {}
        anomaly_metrics = self.last_anomaly_metrics or {}

        prompt = f"""
You are SupplyIQ's Risk Scoring AI.

Your job is to calculate THREE business risks:
1. DEMAND_FORECAST_RISK
2. SUPPLIER_CONTRACT_RISK
3. INVENTORY_RISK

Use the ERP profile, contract profile, anomaly results, and forecast metrics.
Be strict and evidence-based. Do NOT guess from missing data.

Return EXACTLY in this format:

DEMAND_FORECAST_RISK: LOW or MEDIUM or HIGH
SUPPLIER_CONTRACT_RISK: LOW or MEDIUM or HIGH
INVENTORY_RISK: LOW or MEDIUM or HIGH

DEMAND_REASON: <one sentence>
SUPPLIER_CONTRACT_REASON: <one sentence>
INVENTORY_REASON: <one sentence>

DEMAND_EVIDENCE: <short evidence line>
SUPPLIER_CONTRACT_EVIDENCE: <short evidence line>
INVENTORY_EVIDENCE: <short evidence line>

ERP / DEMAND CONTEXT:
{data_profile}

FORECAST METRICS:
{forecast_metrics}

ANOMALY METRICS:
{anomaly_metrics}

CONTRACT CONTEXT:
{contract_profile}
""".strip()

        result = self._llm_generate(prompt)

        def extract(label: str, default: str = "MEDIUM") -> str:
            match = re.search(rf"{label}:\s*(LOW|MEDIUM|HIGH)", result, re.IGNORECASE)
            return match.group(1).upper() if match else default

        def extract_text(label: str, default: str) -> str:
            match = re.search(rf"{label}:\s*(.+)", result)
            return match.group(1).strip() if match else default

        snapshot = {
            "demand_risk": extract("DEMAND_FORECAST_RISK"),
            "contract_risk": extract("SUPPLIER_CONTRACT_RISK"),
            "inventory_risk": extract("INVENTORY_RISK"),
            "demand_reason": extract_text("DEMAND_REASON", "AI demand risk reasoning unavailable."),
            "contract_reason": extract_text("SUPPLIER_CONTRACT_REASON", "AI contract risk reasoning unavailable."),
            "inventory_reason": extract_text("INVENTORY_REASON", "AI inventory risk reasoning unavailable."),
            "demand_evidence": extract_text("DEMAND_EVIDENCE", "ERP demand profile + forecast/anomaly metrics"),
            "contract_evidence": extract_text("SUPPLIER_CONTRACT_EVIDENCE", "Contract clauses and supplier obligations"),
            "inventory_evidence": extract_text("INVENTORY_EVIDENCE", "ERP demand profile + forecast/anomaly metrics"),
            "raw_response": result,
        }

        self.last_ai_risk_snapshot = snapshot
        return snapshot

    def get_kpi_dashboard(self) -> str:
        snapshot = self._get_ai_risk_snapshot()

        demand_risk = snapshot.get("demand_risk", "NOT RUN")
        contract_risk = snapshot.get("contract_risk", "NOT RUN")
        inventory_risk = snapshot.get("inventory_risk", "NOT RUN")

        return self._build_kpi_html(demand_risk, contract_risk, inventory_risk)

    # =========================================================
    # CSV / ERP
    # =========================================================
    def load_erp_data(self, file_path: str) -> str:
        df = None
        last_error = None

        for encoding in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except Exception as e:
                last_error = e

        if df is None:
            raise ValueError(f"Could not read CSV file: {last_error}")

        df.columns = [str(c).strip() for c in df.columns]
        cols_lower = {c.lower(): c for c in df.columns}

        date_col = None
        product_col = None
        demand_col = None

        date_candidates = ["invoicedate", "date", "orderdate", "timestamp", "ds"]
        product_candidates = ["stockcode", "product", "product_code", "item", "item_id", "sku"]
        demand_candidates = ["quantity", "order_demand", "sales", "demand", "y", "units"]

        for c in date_candidates:
            if c in cols_lower:
                date_col = cols_lower[c]
                break

        for c in product_candidates:
            if c in cols_lower:
                product_col = cols_lower[c]
                break

        for c in demand_candidates:
            if c in cols_lower:
                demand_col = cols_lower[c]
                break

        if date_col is None or demand_col is None:
            raise ValueError(
                f"Could not detect required columns. Found columns: {list(df.columns)}"
            )

        if product_col is None:
            df["__product__"] = "ALL_PRODUCTS"
            product_col = "__product__"

        work = df[[date_col, product_col, demand_col]].copy()
        work.columns = ["ds", "product", "y"]

        work["ds"] = pd.to_datetime(work["ds"], errors="coerce")
        work["y"] = pd.to_numeric(work["y"], errors="coerce")

        work = work.dropna(subset=["ds", "y"])
        work = work[work["y"] > 0]

        if work.empty:
            raise ValueError("No valid date/demand rows found after parsing the CSV.")

        daily = work.groupby(work["ds"].dt.date)["y"].sum().reset_index()
        daily["ds"] = pd.to_datetime(daily["ds"])
        daily = daily.sort_values("ds").reset_index(drop=True)

        daily["delivery_delay_days"] = np.random.exponential(scale=2.0, size=len(daily))
        daily["component_price"] = 50 + np.random.normal(0, 4, len(daily))

        self.df = daily.copy()
        self.last_forecast_summary = None
        self.last_forecast_records = None
        self.last_forecast_metrics = None
        self.last_anomaly_summary = None
        self.last_anomaly_records = None
        self.last_anomaly_metrics = None
        self.last_decision_report = None
        self.last_ai_risk_snapshot = None

        if len(self.df) < 15:
            return (
                f"ERP data loaded with {len(self.df)} daily rows. "
                "Upload worked, but 15+ rows are recommended for stronger forecasting."
            )

        return (
            f"ERP data successfully loaded with {len(self.df)} daily rows. "
            "SupplyIQ is ready for demand forecasting and risk detection."
        )

    # =========================================================
    # Forecasting
    # =========================================================
    def run_forecast(self, periods: int = 30):
        if self.df is None:
            return None, "Error: Please upload ERP data first.", None, {}

        train_df = self.df[["ds", "y"]].copy()

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(train_df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        actual = train_df["y"].replace(0, np.nan)
        predicted = forecast["yhat"].iloc[: len(train_df)]
        mape = np.nanmean(np.abs((actual - predicted) / actual)) * 100

        fig = model.plot(forecast)

        upcoming = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()

        avg_demand = float(upcoming["yhat"].mean())
        peak_demand = float(upcoming["yhat"].max())
        volatility_pct = float((upcoming["yhat"].std() / max(avg_demand, 1)) * 100)
        recommended_inventory = int(np.ceil(peak_demand * 1.10))
        peak_to_avg_ratio = peak_demand / max(avg_demand, 1)

        if volatility_pct > 45 or mape > 80:
            risk = "HIGH"
            reason = "Very high demand volatility or unstable historical fit detected."
        elif volatility_pct > 25 or mape > 35:
            risk = "MEDIUM"
            reason = "Moderate volatility detected in historical demand behavior."
        else:
            risk = "LOW"
            reason = "Demand pattern appears comparatively stable."

        summary = f"""
## AI Forecast Insight

**Expected average daily demand:** {avg_demand:.1f} units  
**Peak predicted demand:** {peak_demand:.1f} units  
**Recommended inventory buffer:** {recommended_inventory} units  
**Historical MAPE:** {0 if np.isnan(mape) else mape:.2f}%  

**Demand Forecast Risk:** {self._risk_badge(risk)}  
**Reason:** {reason}
""".strip()

        self.last_forecast_summary = summary
        self.last_forecast_records = upcoming.to_dict(orient="records")
        self.last_forecast_metrics = {
            "avg_demand": avg_demand,
            "peak_demand": peak_demand,
            "mape": 0 if np.isnan(mape) else float(mape),
            "recommended_inventory": recommended_inventory,
            "risk_level": risk,
            "reason": reason,
            "volatility_pct": volatility_pct,
            "peak_to_avg_ratio": peak_to_avg_ratio,
        }
        self.last_ai_risk_snapshot = None

        return fig, summary, self.last_forecast_records, self.last_forecast_metrics

    # =========================================================
    # Anomaly Detection
    # =========================================================
    def detect_anomalies(self) -> Tuple[str, List[Dict], Dict]:
        if self.df is None:
            return "Error: Please upload ERP data first.", [], {}

        features = self.df[["y", "delivery_delay_days", "component_price"]].copy()

        model = IsolationForest(contamination=0.07, random_state=42)
        self.df["anomaly"] = model.fit_predict(features)

        anomalies = self.df[self.df["anomaly"] == -1].tail(6).copy()

        avg_demand = self.df["y"].mean()
        avg_delay = self.df["delivery_delay_days"].mean()
        avg_price = self.df["component_price"].mean()

        lines = []
        records = []

        for _, row in anomalies.iterrows():
            reasons = []

            if row["y"] > avg_demand * 1.35:
                reasons.append("demand spike")
            elif row["y"] < avg_demand * 0.65:
                reasons.append("demand drop")

            if row["delivery_delay_days"] > avg_delay * 1.5:
                reasons.append("high delivery delay")

            if row["component_price"] > avg_price * 1.18:
                reasons.append("component price spike")
            elif row["component_price"] < avg_price * 0.82:
                reasons.append("component price drop")

            reason_text = ", ".join(reasons) if reasons else "unusual multi-factor supply pattern"

            line = (
                f"• **{row['ds'].strftime('%Y-%m-%d')}** — "
                f"Demand={row['y']:.1f}, Delay={row['delivery_delay_days']:.1f} days, "
                f"Price=${row['component_price']:.2f} → {reason_text}"
            )
            lines.append(line)

            records.append(
                {
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "demand": float(row["y"]),
                    "delivery_delay_days": float(row["delivery_delay_days"]),
                    "component_price": float(row["component_price"]),
                    "reason": reason_text,
                }
            )

        anomaly_rate = (len(anomalies) / max(len(self.df), 1)) * 100

        if len(anomalies) >= 5 or anomaly_rate > 8:
            risk = "HIGH"
            headline = "Multiple unusual demand and supply behavior patterns detected."
        elif len(anomalies) >= 2:
            risk = "MEDIUM"
            headline = "Some operational irregularities were detected in recent data."
        else:
            risk = "LOW"
            headline = "Only limited anomaly activity detected in current data."

        if not lines:
            lines = ["• No major anomalies detected in the uploaded ERP data."]

        metrics = {
            "risk_level": risk,
            "headline": headline,
            "anomaly_count": len(anomalies),
            "anomaly_rate_pct": anomaly_rate,
        }

        self.last_anomaly_metrics = metrics
        self.last_ai_risk_snapshot = None
        snapshot = self._get_ai_risk_snapshot()

        final_risk = snapshot.get("demand_risk", risk)
        final_reason = snapshot.get("demand_reason", headline)
        final_evidence = snapshot.get(
            "demand_evidence",
            f"Anomaly count={len(anomalies)}, anomaly rate={anomaly_rate:.2f}%"
        )

        summary = f"""
## Demand Risk Detection Report

**Risk Level:** {self._risk_badge(final_risk)}  
**Why:** {final_reason}  
**Evidence:** {final_evidence}

### Detected Events
{chr(10).join(lines)}
""".strip()

        metrics["risk_level"] = final_risk
        metrics["headline"] = final_reason
        metrics["evidence"] = final_evidence

        self.last_anomaly_summary = summary
        self.last_anomaly_records = records

        return summary, records, metrics

    # =========================================================
    # Document / Contract Intelligence
    # =========================================================
    def setup_document_intelligence(self, pdf_file_path: Optional[str] = None) -> str:
        if not pdf_file_path:
            return "Error: No contract or supplier PDF was provided."

        self.doc_filename = pdf_file_path.split("/")[-1]

        reader = PdfReader(pdf_file_path)
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append(f"[Page {i + 1}] {text}")

        if not pages:
            return "Error: No readable text could be extracted from the PDF."

        full_text = "\n".join(pages)
        self.doc_chunks = self._chunk_text(full_text, chunk_size=1400, overlap=200)

        if not self.doc_chunks:
            return "Error: Failed to create retrievable chunks from the PDF."

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_chunks)
        self.last_contract_report = None
        self.last_contract_risk = "UNKNOWN"
        self.last_decision_report = None
        self.last_ai_risk_snapshot = None

        return (
            f"Contract indexed successfully from '{self.doc_filename}'. "
            f"{len(self.doc_chunks)} retrievable sections created."
        )

    def generate_contract_report(self) -> str:
        if not self.doc_chunks or self.vectorizer is None or self.doc_matrix is None:
            return "Knowledge base not initialized. Please upload a supplier contract PDF first."

        context = self._retrieve_relevant_context(
            "forecast purchase order payment terms delivery penalties manufacturing capacity supplier obligations customer obligations inventory underutilization procurement lead times risk"
        )

        if not context:
            return "No relevant contract content was retrieved."

        prompt = f"""
You are SupplyIQ's Contract Intelligence Agent.

Analyze the supplier/manufacturing contract and return a structured executive report.

Use the contract context below and infer business meaning when reasonable.
Do not say "not explicitly mentioned" unless truly impossible to infer.

Return the answer EXACTLY in this format:

# Contract Intelligence Report

## Customer Obligations
- ...
- ...

## Supplier / Manufacturer Rights
- ...
- ...

## Financial Terms
- ...
- ...

## Operational Constraints
- ...
- ...

## Supply Chain Risks
- ...
- ...

## Executive Takeaway
<2-3 sentence summary>

Also end with:
CONTRACT_RISK_LEVEL: LOW or MEDIUM or HIGH

Contract context:
{context}
""".strip()

        report = self._llm_generate(prompt)

        risk_match = re.search(r"CONTRACT_RISK_LEVEL:\s*(LOW|MEDIUM|HIGH)", report, re.IGNORECASE)
        if risk_match:
            self.last_contract_risk = risk_match.group(1).upper()
            report = re.sub(r"\n*CONTRACT_RISK_LEVEL:\s*(LOW|MEDIUM|HIGH)\s*$", "", report, flags=re.IGNORECASE).strip()
        else:
            self.last_contract_risk = "MEDIUM"

        self.last_contract_report = report
        self.last_ai_risk_snapshot = None
        snapshot = self._get_ai_risk_snapshot()

        final_contract_risk = snapshot.get("contract_risk", self.last_contract_risk)
        final_contract_reason = snapshot.get("contract_reason", "AI contract reasoning unavailable.")
        final_contract_evidence = snapshot.get("contract_evidence", "Contract clauses and obligations")

        report = f"""
{report}

## Final AI Contract Risk
**Risk Level:** {self._risk_badge(final_contract_risk)}  
**Why:** {final_contract_reason}  
**Evidence:** {final_contract_evidence}
""".strip()

        self.last_contract_report = report
        return report

    def query_documents(self, query: str) -> str:
        if not self.doc_chunks or self.vectorizer is None or self.doc_matrix is None:
            return "Knowledge base not initialized. Please upload a PDF first."

        if not query or not query.strip():
            return "Please enter a contract question."

        context = self._retrieve_relevant_context(query, top_k=4)
        if not context:
            return "No relevant content found in the uploaded PDF."

        prompt = f"""
You are SupplyIQ's Contract Intelligence Agent.

Answer the user's question using the contract context below.
Be concise, useful, and business-focused.
If the answer is partly implied, explain the likely implication clearly.

Contract context:
{context}

Question:
{query}
""".strip()

        return self._llm_generate(prompt)

    # =========================================================
    # Decision Engine
    # =========================================================
    def generate_decision(self) -> str:
        if self.df is None:
            return "Error: Please upload ERP data first."
        if not self.doc_chunks or self.vectorizer is None or self.doc_matrix is None:
            return "Error: Please upload and index a supplier contract PDF first."

        if not self.last_forecast_summary:
            self.run_forecast(periods=30)

        if not self.last_anomaly_summary:
            self.detect_anomalies()

        if not self.last_contract_report:
            self.generate_contract_report()

        prompt = f"""
You are the final Supply Chain Decision Engine inside SupplyIQ.

Your task is to produce a VP-level decision report by combining:
1. demand forecasting
2. demand risk detection
3. contract intelligence

Forecast Insight:
{self.last_forecast_summary}

Demand Risk Detection:
{self.last_anomaly_summary}

Contract Intelligence:
{self.last_contract_report}

Return the answer EXACTLY in this format:

# SupplyIQ Decision Report

## Executive Situation
<2-3 sentences>

## What the Forecast Suggests
- ...
- ...

## What the Contract Requires / Allows
- ...
- ...

## Recommended Action
- ...
- ...
- ...

## Business Impact
- ...
- ...

## Final Recommendation
<one strong final recommendation paragraph>
""".strip()

        self.last_decision_report = self._llm_generate(prompt)
        return self.last_decision_report

    # =========================================================
    # Scenario Simulator
    # =========================================================
    def run_scenario(self, scenario_name: str) -> str:
        if self.df is None:
            return "Error: Please upload ERP data first."
        if not self.doc_chunks or self.vectorizer is None or self.doc_matrix is None:
            return "Error: Please upload a contract PDF first."

        if not self.last_forecast_summary:
            self.run_forecast(periods=30)
        if not self.last_anomaly_summary:
            self.detect_anomalies()
        if not self.last_contract_report:
            self.generate_contract_report()

        scenario_map = {
            "Demand increases by 30%": "Demand suddenly rises by 30% over the next month.",
            "Supplier delay increases by 10 days": "Supplier lead time or delivery delay increases by 10 days.",
            "Component price rises by 15%": "Critical component prices increase by 15% unexpectedly.",
            "Forecast error causes over-ordering": "The committed order quantity significantly exceeds real demand.",
        }

        scenario_text = scenario_map.get(scenario_name, scenario_name)

        prompt = f"""
You are SupplyIQ's scenario simulation engine.

Use the business context below and evaluate the scenario.

Forecast Insight:
{self.last_forecast_summary}

Demand Risk Detection:
{self.last_anomaly_summary}

Contract Intelligence:
{self.last_contract_report}

Scenario:
{scenario_text}

Return the answer EXACTLY in this format:

# Scenario Simulation Report

## Scenario
<one sentence>

## Likely Impact
- ...
- ...
- ...

## Main Risks
- ...
- ...

## Recommended Response
- ...
- ...
- ...

## Executive Advice
<short paragraph>
""".strip()

        return self._llm_generate(prompt)
