import { useState, useMemo } from "react";
import { Zap, X, ShieldAlert, BarChart3 } from "lucide-react";

export default function SidePanel({ sessionData, analysisResult, selectedData, onClose }) {
  const [loading, setLoading] = useState(false);
  const [livePrediction, setLivePrediction] = useState(null);

  const { row, shap } = selectedData;
  const { top_features } = analysisResult;

  // Render top SHAP values clearly
  const sortedShap = useMemo(() => {
    return Object.entries(shap || {})
      .map(([k, v]) => ({ feature: k, value: Number(v) }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 5); // top 5 contributors
  }, [shap]);

  const sensitiveCol = sessionData.sensitive_column || analysisResult.detected_sensitive_cols?.[0];

  const handleFlipCounterfactual = async () => {
    if (!sensitiveCol) return;
    setLoading(true);

    try {
      // Artificially mutate the exact demographic record locally
      const flippedRow = { ...row };
      const currentVal = String(flippedRow[sensitiveCol]);
      let targetFlipVal = "0";

      // Simple binary flip heuristic
      if (sessionData.privileged_value && currentVal === String(sessionData.privileged_value)) {
        targetFlipVal = "0"; // To unprivileged
      } else {
        targetFlipVal = sessionData.privileged_value || "1"; // To privileged
      }

      flippedRow[sensitiveCol] = targetFlipVal;

      const formData = new FormData();
      formData.append("model", sessionData.model);
      formData.append("row_data", JSON.stringify(flippedRow));

      // Re-use ordered features from SHAP keys to guarantee correct dimensional layout
      const featureNames = analysisResult.top_features.map(f => f.feature);
      const featureKeys = Object.keys(row).filter(k => k !== sessionData.target_column);
      
      formData.append("feature_cols", JSON.stringify(featureKeys));
      formData.append("target_column", sessionData.target_column);

      const res = await fetch("http://localhost:8000/api/predict-counterfactual", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed");

      setLivePrediction(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: "12px", border: "1px solid var(--border)", height: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "1rem", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0, fontSize: "1.1rem", display: "flex", alignItems: "center", gap: "8px" }}>
          <ShieldAlert size={18} className="icon-accent" /> Local Inspector
        </h3>
        <button className="btn-secondary" style={{ padding: "0.25rem 0.5rem" }} onClick={onClose}><X size={16} /></button>
      </div>

      <div style={{ padding: "1rem", flex: 1, overflowY: "auto" }}>
        
        {/* SHAP attributions */}
        <div>
          <h4 style={{ fontSize: "0.9rem", color: "var(--text-muted)", marginTop: 0, textTransform: "uppercase" }}>Key Drivers</h4>
          {sortedShap.map((s) => (
            <div key={s.feature} style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem", fontSize: "0.9rem", alignItems: "center" }}>
              <span style={{ fontWeight: 500, truncate: true, maxWidth: "120px" }}>{s.feature}</span>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ 
                  width: "100px", height: "6px", background: "rgba(255,255,255,0.1)", borderRadius: "3px", position: "relative" 
                }}>
                  <div style={{
                    position: "absolute",
                    left: "50%",
                    top: 0,
                    height: "100%",
                    width: `${Math.min(50, Math.abs(s.value) * 100)}%`,
                    background: s.value > 0 ? "var(--success)" : "var(--danger)",
                    borderRadius: "3px",
                    transform: s.value > 0 ? "none" : "scaleX(-1)",
                    transformOrigin: "left center" // Visual SHAP bar hack
                  }} />
                </div>
                <span style={{ fontSize: "0.8rem", width: "40px", textAlign: "right", color: s.value > 0 ? "var(--success)" : "var(--danger)" }}>
                  {s.value > 0 ? "+" : ""}{s.value.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </div>

        <hr style={{ borderColor: "rgba(255,255,255,0.05)", margin: "1.5rem 0" }} />

        {/* Counterfactual Lab */}
        <div>
          <h4 style={{ fontSize: "0.9rem", color: "var(--text-muted)", marginTop: 0, textTransform: "uppercase", display: "flex", alignItems: "center", gap: "6px" }}>
            <Zap size={14} className="icon-accent" /> Counterfactual Lab
          </h4>
          
          <div style={{ fontSize: "0.85rem", color: "var(--text-main)", marginBottom: "1rem", lineHeight: 1.5 }}>
            Flipping <code style={{ margin: "0 4px" }}>{sensitiveCol}</code> attribute triggers a live re-prediction vector against the un-mitigated model limits.
          </div>

          <button className="btn-primary" style={{ width: "100%", justifyContent: "center" }} onClick={handleFlipCounterfactual} disabled={loading}>
            {loading ? "Running Vector..." : "Test Flip Hypothesis"}
          </button>

          {livePrediction && (
            <div style={{ marginTop: "1rem", padding: "1rem", background: "rgba(139, 92, 246, 0.1)", borderRadius: "8px", border: "1px solid var(--accent-light)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Predicted Class</span>
                <span style={{ fontWeight: 700, fontSize: "1.2rem", color: String(livePrediction.prediction) === sessionData.positive_label ? "var(--success)" : "var(--danger)" }}>
                  {livePrediction.prediction}
                </span>
              </div>
              {livePrediction.probability !== null && (
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Probability</span>
                  <span style={{ fontWeight: 600 }}>{(livePrediction.probability * 100).toFixed(1)}%</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
