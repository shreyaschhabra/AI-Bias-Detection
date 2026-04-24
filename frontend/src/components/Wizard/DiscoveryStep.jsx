import { useState, useEffect } from "react";
import { ArrowLeft, ArrowRight, Sparkles } from "lucide-react";

export default function DiscoveryStep({ analysisResult, onNext, onPrev }) {
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchSummary() {
      if (!analysisResult) return;
      try {
        const res = await fetch("http://localhost:8000/api/generate-summary", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            metrics: analysisResult.metrics,
            top_features: analysisResult.top_features,
          }),
        });
        const data = await res.json();
        setSummary(data.summary || "Summary generation failed.");
      } catch (e) {
        setSummary("Error reaching LLM API: " + e.message);
      } finally {
        setLoading(false);
      }
    }
    fetchSummary();
  }, [analysisResult]);

  return (
    <div className="glass-panel" style={{ padding: "2.5rem" }}>
      <h2 className="title">Auto-Discovery & Executive Brief</h2>
      <p className="subtitle">
        Operational variables detected. Generating AI insights via Google Gemini.
      </p>

      <div style={{ display: "flex", gap: "2rem", marginTop: "2rem" }}>
        <div style={{ flex: 1 }}>
          <h3 style={{ fontSize: "1.2rem", color: "var(--accent)", marginBottom: "1rem" }}>
            Detected Protected Classes
          </h3>
          <div style={{ background: "rgba(0,0,0,0.3)", padding: "1.5rem", borderRadius: "12px", border: "1px solid var(--border)" }}>
            <ul style={{ paddingLeft: "1.25rem", margin: 0, color: "var(--text-main)", lineHeight: "1.8" }}>
              {analysisResult?.detected_sensitive_cols?.map(col => (
                <li key={col} style={{ fontWeight: 600 }}>{col}</li>
              ))}
            </ul>
          </div>
        </div>

        <div style={{ flex: 2 }}>
          <h3 style={{ fontSize: "1.2rem", color: "var(--accent)", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "8px" }}>
            <Sparkles size={18} /> Executive Summary
          </h3>
          <div style={{ background: "rgba(0,0,0,0.3)", padding: "1.5rem", borderRadius: "12px", border: "1px solid var(--border)", minHeight: "200px" }}>
            {loading ? (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)" }}>
                Generating narrative...
              </div>
            ) : (
              <div style={{ whiteSpace: "pre-wrap", color: "var(--text-main)", lineHeight: "1.6" }}>
                {summary}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="nav-buttons">
        <button className="btn-secondary" onClick={onPrev}><ArrowLeft size={18} /> Back</button>
        <button className="btn-primary" onClick={onNext} disabled={loading}>Continue <ArrowRight size={18} /></button>
      </div>
    </div>
  );
}
