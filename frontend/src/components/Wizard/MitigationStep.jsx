import { useState, useEffect } from "react";
import { CheckCircle2, Download, AlertTriangle } from "lucide-react";

export default function MitigationStep({ sessionData, analysisResult, onPrev }) {
  const [loading, setLoading] = useState(true);
  const [mitigatedData, setMitigatedData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function runMitigation() {
      try {
        const formData = new FormData();
        const sensitiveCol = sessionData.sensitive_column || analysisResult?.detected_sensitive_cols?.[0];
        const privVal = sessionData.privileged_value || "1";

        formData.append("file", sessionData.file);
        formData.append("model", sessionData.model);
        formData.append("target_column", sessionData.target_column);
        formData.append("sensitive_column", sensitiveCol);
        formData.append("privileged_value", privVal);
        formData.append("positive_label", sessionData.positive_label);

        const res = await fetch("http://localhost:8000/api/mitigate", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Mitigation failed");

        setMitigatedData(data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }

    // Only run if we actually have the required columns explicitly defined or auto-detected
    const sensitiveCol = sessionData.sensitive_column || analysisResult?.detected_sensitive_cols?.[0];
    if (sensitiveCol) {
      runMitigation();
    } else {
      setError("Waiting for sensitive column parameters to trigger mitigation.");
      setLoading(false);
    }
  }, [sessionData, analysisResult]);

  const handleDownload = () => {
    if (!mitigatedData?.mitigated_model_base64) return;
    
    // Convert base64 to Blob
    const byteCharacters = atob(mitigatedData.mitigated_model_base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "application/octet-stream" });
    
    // Trigger download
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = "mitigated_model.pkl";
    link.click();
  };

  const MetricComparison = ({ title, before, after, type }) => {
    const isImproved = type === "ratio" 
      ? Math.abs(1 - after) < Math.abs(1 - before)
      : Math.abs(after) < Math.abs(before);

    return (
      <div style={{ display: "flex", justifyContent: "space-between", padding: "1rem", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <span style={{ color: "var(--text-muted)", flex: 1 }}>{title}</span>
        <span style={{ width: "80px", textAlign: "right", color: "var(--danger)" }}>{before.toFixed(3)}</span>
        <span style={{ width: "80px", textAlign: "right", color: isImproved ? "var(--success)" : "var(--text-main)", fontWeight: isImproved ? 700 : 400 }}>
          {after.toFixed(3)}
        </span>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="glass-panel" style={{ padding: "4rem", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
        <div style={{ width: "40px", height: "40px", borderRadius: "50%", border: "4px solid rgba(139, 92, 246, 0.2)", borderTopColor: "var(--accent)", animation: "spin 1s linear infinite" }} />
        <h2 className="title" style={{ fontSize: "1.5rem" }}>Synthesizing Parameters...</h2>
        <p className="subtitle" style={{ margin: 0 }}>Applying Reweighing matrix to the scikit-learn tree structure limits.</p>
        <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  return (
    <div className="glass-panel" style={{ padding: "2.5rem" }}>
      <h2 className="title" style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <CheckCircle2 className="icon-accent" size={32} /> Remediation Dashboard
      </h2>
      <p className="subtitle">
        Your model has been algorithmically constrained using AIF360 Reweighing pre-processing to lift Disparate bounds.
      </p>

      {error ? (
        <div style={{ padding: "2rem", background: "rgba(239, 68, 68, 0.1)", border: "1px solid var(--danger)", borderRadius: "12px", color: "var(--text-main)" }}>
          <AlertTriangle color="var(--danger)" size={24} style={{ marginBottom: "1rem" }} />
          <h3 style={{ margin: "0 0 0.5rem", fontSize: "1.2rem", fontWeight: 600 }}>Mitigation Failed</h3>
          <p>{error}</p>
        </div>
      ) : (
        <div style={{ display: "flex", gap: "2rem", marginTop: "2rem" }}>
          
          <div style={{ flex: 1, background: "rgba(0,0,0,0.3)", borderRadius: "12px", border: "1px solid var(--border)", overflow: "hidden" }}>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "1rem", background: "rgba(255,255,255,0.05)", fontWeight: 600 }}>
              <span style={{ flex: 1 }}>Metric Limit</span>
              <span style={{ width: "80px", textAlign: "right" }}>Before</span>
              <span style={{ width: "80px", textAlign: "right" }}>After</span>
            </div>
            
            <MetricComparison 
              title="Disparate Impact" 
              before={mitigatedData.before_metrics.disparate_impact} 
              after={mitigatedData.after_metrics.disparate_impact} 
              type="ratio" 
            />
            <MetricComparison 
              title="Statistical Parity" 
              before={mitigatedData.before_metrics.statistical_parity_difference} 
              after={mitigatedData.after_metrics.statistical_parity_difference} 
              type="diff" 
            />
            <MetricComparison 
              title="Equal Opportunity" 
              before={mitigatedData.before_metrics.equal_opportunity_difference} 
              after={mitigatedData.after_metrics.equal_opportunity_difference} 
              type="diff" 
            />
          </div>

          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1rem", justifyContent: "center", alignItems: "center", padding: "2rem", background: "linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(16, 185, 129, 0.05))", borderRadius: "12px", border: "1px solid var(--accent-light)", textAlign: "center" }}>
            <div style={{ width: "64px", height: "64px", background: "var(--success)", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: "1rem", boxShadow: "0 4px 20px rgba(16, 185, 129, 0.4)" }}>
              <CheckCircle2 color="white" size={32} />
            </div>
            
            <h3 style={{ margin: 0, fontSize: "1.5rem", color: "var(--text-main)" }}>Fair Model Ready</h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", lineHeight: 1.5 }}>
              The mathematical distribution vectors have stabilized. The `.pkl` format represents an in-place architecture upgrade mapped cleanly to your original schema limits.
            </p>

            <button className="btn-primary" style={{ marginTop: "1rem", width: "100%", justifyContent: "center" }} onClick={handleDownload}>
              <Download size={20} /> Download Re-Weighed Model .pkl
            </button>
          </div>

        </div>
      )}

      <div className="nav-buttons">
        <button className="btn-secondary" onClick={onPrev}>Back to Explorer</button>
      </div>
    </div>
  );
}
