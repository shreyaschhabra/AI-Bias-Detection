import { useState } from "react";
import { UploadCloud } from "lucide-react";

export default function UploadStep({ sessionData, setSessionData, onNext, setAnalysisResult }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e, field) => {
    setSessionData((prev) => ({ ...prev, [field]: e.target.files[0] }));
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", sessionData.file);
    formData.append("model", sessionData.model);
    formData.append("target_column", sessionData.target_column);
    if (sessionData.sensitive_column) {
      formData.append("sensitive_column", sessionData.sensitive_column);
    }
    if (sessionData.privileged_value) {
      formData.append("privileged_value", sessionData.privileged_value);
    }
    formData.append("positive_label", sessionData.positive_label);

    try {
      const res = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Analysis failed");

      setAnalysisResult(data);
      onNext();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isFormValid = sessionData.file && sessionData.model && sessionData.target_column;

  return (
    <div className="glass-panel p-8" style={{ padding: "2rem" }}>
      <h2 className="title">Data & Model Upload</h2>
      <p className="subtitle">Please provide your operational dataset and trained scikit-learn model.</p>

      {error && <div style={{ color: "var(--danger)", marginBottom: "1rem" }}>{error}</div>}

      <div style={{ display: "flex", gap: "2rem" }}>
        <div style={{ flex: 1 }}>
          <div className="form-group">
            <label>Dataset (CSV)</label>
            <input type="file" accept=".csv" onChange={(e) => handleFileChange(e, "file")} className="form-input" />
          </div>
          <div className="form-group">
            <label>Sklearn Model (.pkl)</label>
            <input type="file" accept=".pkl" onChange={(e) => handleFileChange(e, "model")} className="form-input" />
          </div>
        </div>

        <div style={{ flex: 1 }}>
          <div className="form-group">
            <label>Target Column</label>
            <input 
              type="text" 
              className="form-input" 
              placeholder="e.g. income" 
              value={sessionData.target_column} 
              onChange={(e) => setSessionData(prev => ({...prev, target_column: e.target.value}))} 
            />
          </div>
          <div className="form-group">
            <label>Positive Label</label>
            <input 
              type="text" 
              className="form-input" 
              placeholder="e.g. 1 or Yes" 
              value={sessionData.positive_label} 
              onChange={(e) => setSessionData(prev => ({...prev, positive_label: e.target.value}))} 
            />
          </div>
          <div className="form-group">
            <label>Sensitive Column (Optional - Auto-Detected)</label>
            <input 
              type="text" 
              className="form-input" 
              placeholder="e.g. sex" 
              value={sessionData.sensitive_column} 
              onChange={(e) => setSessionData(prev => ({...prev, sensitive_column: e.target.value}))} 
            />
          </div>
          <div className="form-group">
            <label>Privileged Value (Optional - Auto-Detected)</label>
            <input 
              type="text" 
              className="form-input" 
              placeholder="e.g. 1" 
              value={sessionData.privileged_value} 
              onChange={(e) => setSessionData(prev => ({...prev, privileged_value: e.target.value}))} 
            />
          </div>
        </div>
      </div>

      <div className="nav-buttons">
        <button className="btn-primary" onClick={handleAnalyze} disabled={!isFormValid || loading}>
          {loading ? "Analyzing..." : "Analyze Model Bias" } <UploadCloud size={20} />
        </button>
      </div>
    </div>
  );
}
