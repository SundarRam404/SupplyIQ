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

    def _estimate_inventory_risk(self) -> str:
        if self.last_forecast_metrics is None:
            return "UNKNOWN"

        volatility = self.last_forecast_metrics.get("volatility_pct", 0)
        peak_ratio = self.last_forecast_metrics.get("peak_to_avg_ratio", 1.0)

        if volatility > 45 or peak_ratio > 1.65:
            return "HIGH"
        if volatility > 25 or peak_ratio > 1.35:
            return "MEDIUM"
        return "LOW"

    def _build_kpi_html(self, demand_risk: str, contract_risk: str, inventory_risk: str) -> str:
        def card(title: str, value: str) -> str:
            color = {
                "LOW": "#22c55e",
                "MEDIUM": "#f59e0b",
                "HIGH": "#ef4444",
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

    def get_kpi_dashboard(self) -> str:
        demand_risk = self.last_anomaly_metrics["risk_level"] if self.last_anomaly_metrics else "UNKNOWN"
        contract_risk = self.last_contract_risk or "UNKNOWN"
        inventory_risk = self._estimate_inventory_risk()
        return self._build_kpi_html(demand_risk, contract_risk, inventory_risk)

    # =========================================================
    # CSV / ERP
    # =========================================================
    def load_erp_data(self, file_path: str) -> str:
        self.df = pd.read_csv(file_path)

        if "ds" not in self.df.columns or "y" not in self.df.columns:
            self.df["ds"] = pd.to_datetime(self.df.iloc[:, 0], errors="coerce")
            self.df["y"] = pd.to_numeric(self.df.iloc[:, 1], errors="coerce")
        else:
            self.df["ds"] = pd.to_datetime(self.df["ds"], errors="coerce")
            self.df["y"] = pd.to_numeric(self.df["y"], errors="coerce")

        if "delivery_delay_days" not in self.df.columns:
            self.df["delivery_delay_days"] = np.random.exponential(scale=2.0, size=len(self.df))
        else:
            self.df["delivery_delay_days"] = pd.to_numeric(
                self.df["delivery_delay_days"], errors="coerce"
            )

        if "component_price" not in self.df.columns:
            self.df["component_price"] = 50 + np.random.normal(0, 4, len(self.df))
        else:
            self.df["component_price"] = pd.to_numeric(
                self.df["component_price"], errors="coerce"
            )

        self.df = self.df.dropna(subset=["ds", "y"]).copy()
        self.df["delivery_delay_days"] = self.df["delivery_delay_days"].fillna(
            self.df["delivery_delay_days"].median()
        )
        self.df["component_price"] = self.df["component_price"].fillna(
            self.df["component_price"].median()
        )
        self.df = self.df.sort_values("ds").reset_index(drop=True)

        if len(self.df) < 15:
            return (
                f"ERP data loaded with {len(self.df)} rows. "
                "Upload worked, but 15+ rows are recommended for stronger forecasting."
            )

        return (
            f"ERP data successfully loaded with {len(self.df)} rows. "
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

        summary = f"""
## Demand Risk Detection Report

**Risk Level:** {self._risk_badge(risk)}  
**Headline:** {headline}

### Detected Events
{chr(10).join(lines)}
""".strip()

        metrics = {
            "risk_level": risk,
            "headline": headline,
            "anomaly_count": len(anomalies),
            "anomaly_rate_pct": anomaly_rate,
        }

        self.last_anomaly_summary = summary
        self.last_anomaly_records = records
        self.last_anomaly_metrics = metrics

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