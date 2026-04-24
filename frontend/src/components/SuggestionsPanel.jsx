export function SuggestionsPanel({ suggestions }) {
  const isGreen = suggestions.length === 1 && suggestions[0].startsWith("No major");
  return (
    <div
      style={{
        background: isGreen ? "#f0fff4" : "#fffaf0",
        border: `1px solid ${isGreen ? "#68d391" : "#f6ad55"}`,
        borderRadius: "6px",
        padding: "14px 18px",
      }}
    >
      <h3 style={{ marginBottom: "10px", fontSize: "17px" }}>Recommendations</h3>
      <ul style={{ margin: 0, paddingLeft: "20px" }}>
        {suggestions.map((s, i) => (
          <li key={i} style={{ marginBottom: "6px", fontSize: "14px", color: "#4a5568", lineHeight: 1.5 }}>
            {s}
          </li>
        ))}
      </ul>
    </div>
  );
}
