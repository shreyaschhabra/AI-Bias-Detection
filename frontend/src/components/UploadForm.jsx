import { useState } from "react";

export function UploadForm({ onSubmit, disabled }) {
  const [csvFile, setCsvFile] = useState(null);
  const [modelFile, setModelFile] = useState(null);
  const [targetCol, setTargetCol] = useState("");
  const [sensitiveCol, setSensitiveCol] = useState("");
  const [privilegedVal, setPrivilegedVal] = useState("");
  const [positiveLabel, setPositiveLabel] = useState("1");
  const [errors, setErrors] = useState({});

  function validate() {
    const e = {};
    if (!csvFile) e.csv = "Dataset CSV is required";
    if (!modelFile) e.model = "Model file is required";
    if (!targetCol.trim()) e.targetCol = "Target column is required";
    if (!sensitiveCol.trim()) e.sensitiveCol = "Sensitive column is required";
    if (!privilegedVal.trim()) e.privilegedVal = "Privileged value is required";
    return e;
  }

  function handleSubmit(e) {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      return;
    }
    setErrors({});
    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("model", modelFile);
    formData.append("target_column", targetCol.trim());
    formData.append("sensitive_column", sensitiveCol.trim());
    formData.append("privileged_value", privilegedVal.trim());
    formData.append("positive_label", positiveLabel.trim() || "1");
    onSubmit(formData);
  }

  const inputStyle = (field) => ({
    width: "100%",
    padding: "8px",
    marginTop: "4px",
    border: `1px solid ${errors[field] ? "#e53e3e" : "#cbd5e0"}`,
    borderRadius: "4px",
    fontSize: "14px",
    boxSizing: "border-box",
  });

  return (
    <form onSubmit={handleSubmit} style={{ maxWidth: "560px" }}>
      <h2 style={{ marginBottom: "20px", fontSize: "20px" }}>Analyze Model Bias</h2>

      <div style={{ marginBottom: "14px" }}>
        <label style={{ fontWeight: 600, fontSize: "14px" }}>Dataset (CSV)</label>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setCsvFile(e.target.files[0])}
          style={inputStyle("csv")}
        />
        {errors.csv && <p style={{ color: "#e53e3e", fontSize: "12px", margin: "2px 0 0" }}>{errors.csv}</p>}
      </div>

      <div style={{ marginBottom: "14px" }}>
        <label style={{ fontWeight: 600, fontSize: "14px" }}>Model File (.pkl)</label>
        <input
          type="file"
          accept=".pkl"
          onChange={(e) => setModelFile(e.target.files[0])}
          style={inputStyle("model")}
        />
        {errors.model && <p style={{ color: "#e53e3e", fontSize: "12px", margin: "2px 0 0" }}>{errors.model}</p>}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "14px" }}>
        <div>
          <label style={{ fontWeight: 600, fontSize: "14px" }}>Target Column</label>
          <input
            type="text"
            placeholder="e.g. income"
            value={targetCol}
            onChange={(e) => setTargetCol(e.target.value)}
            style={inputStyle("targetCol")}
          />
          {errors.targetCol && <p style={{ color: "#e53e3e", fontSize: "12px", margin: "2px 0 0" }}>{errors.targetCol}</p>}
        </div>
        <div>
          <label style={{ fontWeight: 600, fontSize: "14px" }}>Sensitive Column</label>
          <input
            type="text"
            placeholder="e.g. sex"
            value={sensitiveCol}
            onChange={(e) => setSensitiveCol(e.target.value)}
            style={inputStyle("sensitiveCol")}
          />
          {errors.sensitiveCol && <p style={{ color: "#e53e3e", fontSize: "12px", margin: "2px 0 0" }}>{errors.sensitiveCol}</p>}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "20px" }}>
        <div>
          <label style={{ fontWeight: 600, fontSize: "14px" }}>Privileged Value</label>
          <input
            type="text"
            placeholder="e.g. 1 or Male"
            value={privilegedVal}
            onChange={(e) => setPrivilegedVal(e.target.value)}
            style={inputStyle("privilegedVal")}
          />
          {errors.privilegedVal && <p style={{ color: "#e53e3e", fontSize: "12px", margin: "2px 0 0" }}>{errors.privilegedVal}</p>}
        </div>
        <div>
          <label style={{ fontWeight: 600, fontSize: "14px" }}>Positive Label</label>
          <input
            type="text"
            placeholder="e.g. 1"
            value={positiveLabel}
            onChange={(e) => setPositiveLabel(e.target.value)}
            style={inputStyle("positiveLabel")}
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={disabled}
        style={{
          padding: "10px 24px",
          background: disabled ? "#a0aec0" : "#3182ce",
          color: "white",
          border: "none",
          borderRadius: "4px",
          fontSize: "15px",
          fontWeight: 600,
          cursor: disabled ? "not-allowed" : "pointer",
          width: "100%",
        }}
      >
        {disabled ? "Analyzing..." : "Analyze"}
      </button>
    </form>
  );
}
