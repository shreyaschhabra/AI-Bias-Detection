const METRIC_CONFIG = {
  statistical_parity_difference: {
    label: "Statistical Parity Difference",
    good: (v) => Math.abs(v) <= 0.1,
    warn: (v) => Math.abs(v) <= 0.2,
    format: (v) => v.toFixed(3),
    tooltip: "|SPD| ≤ 0.1 is fair. Measures difference in positive-outcome rates between groups.",
  },
  disparate_impact: {
    label: "Disparate Impact",
    good: (v) => v >= 0.8,
    warn: (v) => v >= 0.6,
    format: (v) => v.toFixed(3),
    tooltip: "DI ≥ 0.8 meets the 4/5ths legal standard. Ratio of positive rates (unprivileged / privileged).",
  },
  equal_opportunity_difference: {
    label: "Equal Opportunity Difference",
    good: (v) => Math.abs(v) <= 0.1,
    warn: (v) => Math.abs(v) <= 0.2,
    format: (v) => v.toFixed(3),
    tooltip: "|EOD| ≤ 0.1 is fair. Difference in true positive rates between groups.",
  },
  consistency_score: {
    label: "Consistency Score",
    good: (v) => v >= 0.9,
    warn: (v) => v >= 0.7,
    format: (v) => v.toFixed(3),
    tooltip: "Higher is fairer. Measures if similar individuals receive similar predictions.",
  },
  generalized_entropy_error: {
    label: "Generalized Entropy Error",
    good: (v) => v <= 0.2,
    warn: (v) => v <= 0.4,
    format: (v) => v.toFixed(3),
    tooltip: "Lower is fairer. Measures inequality in model benefit distribution.",
  },
};

function statusColor(config, value) {
  if (config.good(value)) return { bg: "#f0fff4", border: "#38a169", badge: "#38a169", label: "Fair" };
  if (config.warn(value)) return { bg: "#fffaf0", border: "#dd6b20", badge: "#dd6b20", label: "Warn" };
  return { bg: "#fff5f5", border: "#e53e3e", badge: "#e53e3e", label: "Flag" };
}

export function MetricsPanel({ metrics }) {
  return (
    <div>
      <h3 style={{ marginBottom: "12px", fontSize: "17px" }}>Fairness Metrics</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "12px" }}>
        {Object.entries(METRIC_CONFIG).map(([key, cfg]) => {
          const value = metrics[key];
          if (value === undefined) return null;
          const colors = statusColor(cfg, value);
          return (
            <div
              key={key}
              title={cfg.tooltip}
              style={{
                background: colors.bg,
                border: `1.5px solid ${colors.border}`,
                borderRadius: "6px",
                padding: "14px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                <span style={{ fontSize: "12px", color: "#718096", fontWeight: 500 }}>{cfg.label}</span>
                <span
                  style={{
                    background: colors.badge,
                    color: "white",
                    borderRadius: "3px",
                    padding: "1px 6px",
                    fontSize: "11px",
                    fontWeight: 700,
                  }}
                >
                  {colors.label}
                </span>
              </div>
              <div style={{ fontSize: "24px", fontWeight: 700, color: "#2d3748" }}>{cfg.format(value)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
