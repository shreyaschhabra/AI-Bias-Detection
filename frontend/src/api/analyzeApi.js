export async function runAnalysis(formData) {
  const response = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    let detail = "Analysis failed";
    try {
      const err = await response.json();
      detail = err.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}
