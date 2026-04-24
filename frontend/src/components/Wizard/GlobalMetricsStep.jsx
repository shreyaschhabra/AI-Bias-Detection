import { ArrowLeft, ArrowRight, BarChart2 } from "lucide-react";

export default function GlobalMetricsStep({ analysisResult, onNext, onPrev }) {
  if (!analysisResult) return null;

  const { metrics, group_selection_rates, intersectional_metrics } = analysisResult;

  const MetricCard = ({ title, value, type }) => {
    let colorClass = "var(--text-main)";
    let desc = "";
    
    // Simplistic threshold checks
    if (type === "ratio") {
      const v = Math.abs(1 - value);
      colorClass = v > 0.2 ? "var(--danger)" : "var(--success)";
      desc = "Ideal: 1.0 (80% Rule)";
    } else if (type === "diff") {
      const v = Math.abs(value);
      colorClass = v > 0.1 ? "var(--danger)" : "var(--success)";
      desc = "Ideal: 0.0";
    }

    return (
      <div className="glass-panel" style={{ padding: "1.5rem", borderTop: `4px solid ${colorClass}` }}>
        <h4 style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: 0 }}>{title}</h4>
        <div style={{ fontSize: "2rem", fontWeight: 700, color: colorClass, margin: "0.5rem 0" }}>
          {value.toFixed(3)}
        </div>
        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>{desc}</div>
      </div>
    );
  };

  return (
    <div className="glass-panel" style={{ padding: "2.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "1rem" }}>
        <BarChart2 className="icon-accent" size={28} />
        <h2 className="title" style={{ margin: 0 }}>Global Fairness Metrics</h2>
      </div>
      <p className="subtitle">Mathematical evaluation of overall model bias.</p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1.5rem", marginBottom: "3rem" }}>
        <MetricCard title="Disparate Impact" value={metrics.disparate_impact} type="ratio" />
        <MetricCard title="Statistical Parity Diff" value={metrics.statistical_parity_difference} type="diff" />
        <MetricCard title="Equal Opportunity Diff" value={metrics.equal_opportunity_difference} type="diff" />
        <MetricCard title="Consistency Score" value={metrics.consistency_score} type="ratio" />
      </div>

      <div style={{ display: "flex", gap: "2rem", marginBottom: "2rem" }}>
        <div style={{ flex: 1, background: "rgba(0,0,0,0.2)", padding: "1.5rem", borderRadius: "12px", border: "1px solid var(--border)" }}>
          <h3 style={{ fontSize: "1.1rem", marginTop: 0, color: "var(--accent)" }}>Group Selection Rates</h3>
          {Object.entries(group_selection_rates).map(([grp, rate]) => (
            <div key={grp} style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem", paddingBottom: "0.5rem", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
              <span style={{ textTransform: "capitalize", color: "var(--text-muted)" }}>{grp}</span>
              <span style={{ fontWeight: 600 }}>{(rate * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>

        <div style={{ flex: 1, background: "rgba(0,0,0,0.2)", padding: "1.5rem", borderRadius: "12px", border: "1px solid var(--border)" }}>
          <h3 style={{ fontSize: "1.1rem", marginTop: 0, color: "var(--accent)" }}>Intersectional Extremes</h3>
          {intersectional_metrics?.slice(0, 3).map((im, idx) => (
            <div key={idx} style={{ marginBottom: "0.75rem", fontSize: "0.9rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-muted)" }}>
                <span>{im.subgroup}</span>
                <span style={{ color: im.disparate_impact < 0.8 ? "var(--warning)" : "var(--text-main)"}}>
                  DI: {im.disparate_impact?.toFixed(2) || "N/A"}
                </span>
              </div>
            </div>
          ))}
          {(!intersectional_metrics || intersectional_metrics.length === 0) && (
            <div style={{ color: "var(--text-muted)", fontStyle: "italic" }}>No intersectional boundaries detected.</div>
          )}
        </div>
      </div>

      <div className="nav-buttons">
        <button className="btn-secondary" onClick={onPrev}><ArrowLeft size={18} /> Back</button>
        <button className="btn-primary" onClick={onNext}>Inspect Local SHAP <ArrowRight size={18} /></button>
      </div>
    </div>
  );
}
