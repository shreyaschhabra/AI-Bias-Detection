function isProxy(f) {
  return f.importance > 0.05 && f.corr_with_sensitive > 0.5;
}

export function FeatureTable({ features }) {
  return (
    <div>
      <h3 style={{ marginBottom: "10px", fontSize: "17px" }}>Feature Importance & Proxy Analysis</h3>
      <p style={{ fontSize: "13px", color: "#718096", marginBottom: "10px" }}>
        Rows highlighted in yellow have high model importance AND strong correlation with the sensitive
        attribute — potential proxy features causing indirect discrimination.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
          <thead>
            <tr style={{ background: "#edf2f7" }}>
              <th style={th}>Feature</th>
              <th style={{ ...th, textAlign: "right" }}>Importance</th>
              <th style={{ ...th, textAlign: "right" }}>Corr. with Sensitive</th>
              <th style={{ ...th, textAlign: "center" }}>Proxy Risk</th>
            </tr>
          </thead>
          <tbody>
            {features.map((f, i) => {
              const proxy = isProxy(f);
              return (
                <tr
                  key={i}
                  style={{ background: proxy ? "#fefcbf" : i % 2 === 0 ? "#fff" : "#f7fafc" }}
                >
                  <td style={td}>{f.feature}</td>
                  <td style={{ ...td, textAlign: "right" }}>{f.importance.toFixed(4)}</td>
                  <td style={{ ...td, textAlign: "right" }}>{f.corr_with_sensitive.toFixed(4)}</td>
                  <td style={{ ...td, textAlign: "center" }}>
                    {proxy ? (
                      <span
                        style={{
                          background: "#dd6b20",
                          color: "white",
                          borderRadius: "3px",
                          padding: "1px 7px",
                          fontSize: "11px",
                          fontWeight: 700,
                        }}
                      >
                        High
                      </span>
                    ) : (
                      <span style={{ color: "#a0aec0", fontSize: "12px" }}>—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const th = {
  padding: "8px 12px",
  textAlign: "left",
  fontWeight: 600,
  fontSize: "13px",
  color: "#4a5568",
  border: "1px solid #e2e8f0",
};

const td = {
  padding: "7px 12px",
  border: "1px solid #e2e8f0",
  color: "#2d3748",
};
