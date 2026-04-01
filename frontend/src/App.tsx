import { useMemo, useState } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

type Metrics = {
  dataset_id: string;
  rows: number;
  features: number;
  label_column: string;
  roc_auc: number;
  precision_fraud: number;
  recall_fraud: number;
  f1_fraud: number;
};

type Prediction = {
  is_fraud: boolean;
  fraud_probability: number;
  threshold: number;
};

function formatPct(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

export default function App() {
  const [datasetId, setDatasetId] = useState("David-Egea/Creditcard-fraud-detection");
  const [health, setHealth] = useState<string>("Unknown");
  const [features, setFeatures] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [featureValues, setFeatureValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const topFeatures = useMemo(() => features.slice(0, 12), [features]);

  async function checkHealth() {
    setError("");
    try {
      const { data } = await axios.get(`${API_URL}/health`);
      setHealth(data.model_loaded ? "API online - model ready" : "API online - model not trained");
    } catch (e) {
      setHealth("API unreachable");
      setError("Cannot connect to backend at http://127.0.0.1:8000");
    }
  }

  async function trainModel() {
    setLoading(true);
    setError("");
    setPrediction(null);
    try {
      const { data } = await axios.post(`${API_URL}/train`, {
        dataset_id: datasetId,
        test_size: 0.2,
        random_state: 42
      });
      setMetrics(data.metrics);
      await loadFeatures();
      await checkHealth();
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Training failed. Check dataset id and backend logs.");
    } finally {
      setLoading(false);
    }
  }

  async function loadFeatures() {
    setError("");
    try {
      const { data } = await axios.get(`${API_URL}/features`);
      setFeatures(data.feature_names);
      const seeded: Record<string, string> = {};
      data.feature_names.slice(0, 12).forEach((name: string) => {
        seeded[name] = "0";
      });
      setFeatureValues(seeded);
    } catch (e: any) {
      const detail = e?.response?.data?.detail;
      setError(
        detail === "Model not trained yet."
          ? "No model yet — click Train Model first, wait for it to finish, then try again."
          : detail || "Could not load model features."
      );
    }
  }

  async function predict() {
    setError("");
    const payload: Record<string, number> = {};
    topFeatures.forEach((name) => {
      const raw = featureValues[name];
      payload[name] = Number(raw || 0);
    });

    try {
      const { data } = await axios.post(`${API_URL}/predict`, { features: payload });
      setPrediction(data);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Prediction failed.");
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>FraudShield AI</h1>
        <p>Portfolio-ready credit card fraud detection platform (FastAPI + React + Hugging Face data).</p>
      </header>

      <section className="card">
        <h2>1) Backend status</h2>
        <div className="row">
          <button onClick={checkHealth}>Check API Health</button>
          <span className="status">{health}</span>
        </div>
      </section>

      <section className="card">
        <h2>2) Train model from Hugging Face</h2>
        <div className="row">
          <input value={datasetId} onChange={(e) => setDatasetId(e.target.value)} placeholder="dataset id" />
          <button onClick={trainModel} disabled={loading}>{loading ? "Training..." : "Train Model"}</button>
          <button onClick={loadFeatures}>Load Features</button>
        </div>
        <small>Default: <code>David-Egea/Creditcard-fraud-detection</code> (classic Kaggle-style creditcard data on the Hub). You can swap to another compatible fraud dataset.</small>
        <p className="hint">
          Until you train once, the API has no model — use <strong>Train Model</strong> first (downloads from Hugging Face; first run may take several minutes). Then use <strong>Load Features</strong> or predict.
        </p>
      </section>

      {metrics && (
        <section className="card metrics">
          <h2>3) Model quality snapshot</h2>
          <div className="grid">
            <div><strong>Rows</strong><span>{metrics.rows}</span></div>
            <div><strong>Features</strong><span>{metrics.features}</span></div>
            <div><strong>ROC-AUC</strong><span>{metrics.roc_auc.toFixed(4)}</span></div>
            <div><strong>Precision (fraud)</strong><span>{formatPct(metrics.precision_fraud)}</span></div>
            <div><strong>Recall (fraud)</strong><span>{formatPct(metrics.recall_fraud)}</span></div>
            <div><strong>F1 (fraud)</strong><span>{formatPct(metrics.f1_fraud)}</span></div>
          </div>
        </section>
      )}

      {topFeatures.length > 0 && (
        <section className="card">
          <h2>4) Simulate a transaction</h2>
          <details className="feature-help">
            <summary>What do V1, V2, V3… mean?</summary>
            <p>
              In this dataset, <strong>V1–V28</strong> are not real business fields (like “merchant” or “country”). They are{" "}
              <strong>anonymized, PCA-transformed</strong> versions of the original transaction variables, published so fraud
              patterns can be studied without exposing private details. You cannot map a single V-column to plain English.
            </p>
            <p>
              <strong>Time</strong> is seconds since the first transaction in the set; <strong>Amount</strong> is the
              transaction amount. Together with V1–V28 they form the numeric signal the model learned on.
            </p>
            <p>
              This screen only edits the <strong>first 12</strong> features to keep the form small; the backend fills any
              other features using the same median imputation as training. For a quick demo, try small changes around{" "}
              <code>0</code> or use <strong>Amount</strong> if it appears in the list — the score shows how the model reacts,
              not a real-world “fraud story.”
            </p>
          </details>
          <p className="simulate-lead">
            Adjust values below (demo / exploration). Missing features are filled with median imputation on the server.
          </p>
          <div className="feature-grid">
            {topFeatures.map((name) => (
              <label key={name}>
                <span>{name}</span>
                <input
                  type="number"
                  step="any"
                  value={featureValues[name] ?? "0"}
                  onChange={(e) =>
                    setFeatureValues((prev) => ({
                      ...prev,
                      [name]: e.target.value
                    }))
                  }
                />
              </label>
            ))}
          </div>
          <button onClick={predict}>Predict Fraud Risk</button>
        </section>
      )}

      {prediction && (
        <section className="card result">
          <h2>Prediction Result</h2>
          <p className={prediction.is_fraud ? "risk high" : "risk low"}>
            {prediction.is_fraud ? "High Risk: Potential Fraud" : "Low Risk: Likely Legitimate"}
          </p>
          <p>Fraud Probability: <strong>{formatPct(prediction.fraud_probability)}</strong></p>
          <p>Decision Threshold: <strong>{prediction.threshold}</strong></p>
        </section>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}
