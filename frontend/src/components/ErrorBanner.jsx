export function ErrorBanner({ message }) {
  return (
    <div
      style={{
        background: "#fff5f5",
        border: "1px solid #fc8181",
        borderRadius: "6px",
        padding: "12px 16px",
        color: "#c53030",
        fontSize: "14px",
      }}
    >
      <strong>Error: </strong>{message}
    </div>
  );
}
