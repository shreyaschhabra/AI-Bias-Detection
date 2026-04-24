export function CounterfactualPanel({ flipRate }) {
  const pct = (flipRate * 100).toFixed(1);
  const isHigh = flipRate > 0.15;
  return (
    <div
      style={{
        background: isHigh ? "#fff5f5" : "#f7fafc",
        border: `1px solid ${isHigh ? "#fc8181" : "#e2e8f0"}`,
        borderRadius: "6px",
        padding: "14px 18px",
      }}
    >
      <h3 style={{ marginBottom: "6px", fontSize: "17px" }}>Counterfactual Test</h3>
      <p style={{ margin: 0, fontSize: "14px", color: "#4a5568" }}>
        <strong style={{ color: isHigh ? "#e53e3e" : "#2d3748" }}>{pct}%</strong> of records changed
        their predicted outcome when the sensitive attribute was toggled.
        {isHigh
          ? " This suggests the model is sensitive to the protected attribute, either directly or through correlated proxy features."
          : " The model appears stable to changes in the sensitive attribute."}
      </p>
    </div>
  );
}
