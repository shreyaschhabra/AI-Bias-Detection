import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from "recharts";

export function SelectionRateChart({ rates }) {
  const data = Object.entries(rates).map(([group, rate]) => ({
    group: group.charAt(0).toUpperCase() + group.slice(1),
    rate: parseFloat((rate * 100).toFixed(1)),
  }));

  const COLORS = ["#3182ce", "#e53e3e"];

  return (
    <div>
      <h3 style={{ marginBottom: "12px", fontSize: "17px" }}>Positive Outcome Rate by Group</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <XAxis dataKey="group" tick={{ fontSize: 13 }} />
          <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(v) => `${v}%`} />
          <Bar dataKey="rate" radius={[4, 4, 0, 0]}>
            {data.map((_, index) => (
              <Cell key={index} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
