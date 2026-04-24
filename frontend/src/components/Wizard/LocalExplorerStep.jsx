import { useState, useEffect, useMemo } from "react";
import { ArrowLeft, ArrowRight, MousePointerClick } from "lucide-react";
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import Papa from "papaparse";
import SidePanel from "./SidePanel";

export default function LocalExplorerStep({ sessionData, analysisResult, onNext, onPrev }) {
  const [dataset, setDataset] = useState([]);
  const [selectedPoint, setSelectedPoint] = useState(null);

  const { shap_values } = analysisResult;

  // Load the original CSV into memory for counterfactual injection
  useEffect(() => {
    if (sessionData.file && shap_values?.length > 0) {
      Papa.parse(sessionData.file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          // Truncate to match SHAP if backend sampled
          const cappedData = results.data.slice(0, shap_values.length);
          setDataset(cappedData);
        }
      });
    }
  }, [sessionData.file, shap_values]);

  // Sub-Sample for Recharts to avoid DOM crashing (e.g. max 400 dots)
  const chartData = useMemo(() => {
    if (!shap_values) return [];
    
    // Create random x-scatter for beeswarm effect
    const data = shap_values.map((shapMap, idx) => {
      // Pick top feature for Y axis distribution
      const topFeature = Object.keys(shapMap)[0]; // heuristic
      
      return {
        id: idx,
        x: Math.random() * 100,
        y: shapMap[topFeature] || 0,
        index: idx,
        isPositive: (shapMap[topFeature] || 0) > 0
      };
    });

    // Sample down to 400 points
    return data.slice(0, 400);
  }, [shap_values]);

  const handlePointClick = (data) => {
    if (!dataset[data.index]) return;
    
    setSelectedPoint({
      row: dataset[data.index],
      shap: shap_values[data.index]
    });
  };

  return (
    <div className="glass-panel" style={{ padding: "2.5rem", position: "relative" }}>
      <h2 className="title">Local Interpretability Explorer</h2>
      <p className="subtitle">
        Visualizing {chartData.length} records. Click any <span style={{color: "var(--accent)"}}>data point</span> to inspect SHAP attributions and execute live counterfactuals.
      </p>

      <div style={{ display: "flex", gap: "2rem", height: "450px" }}>
        
        {/* Scatter Plot */}
        <div style={{ flex: selectedPoint ? 2 : 1, transition: "all 0.3s ease", background: "rgba(0,0,0,0.2)", borderRadius: "12px", border: "1px solid var(--border)", display: "flex", flexDirection: "column" }}>
          
          {!selectedPoint && (
            <div style={{ padding: "1rem", textAlign: "center", color: "var(--text-muted)", fontSize: "0.9rem", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
              <MousePointerClick size={16} /> Select a point to inspect
            </div>
          )}

          <div style={{ flex: 1, width: "100%", padding: "10px" }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <XAxis type="number" dataKey="x" hide />
                <YAxis type="number" dataKey="y" tick={{ fill: "var(--text-muted)" }} domain={['auto', 'auto']} />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }} 
                  contentStyle={{ backgroundColor: "var(--bg-panel)", border: "1px solid var(--border)", borderRadius: "8px", backdropFilter: "blur(8px)" }}
                />
                <Scatter name="SHAP distribution" data={chartData} onClick={handlePointClick} style={{ cursor: "pointer" }}>
                  {chartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.id === selectedPoint?.index ? "#ffffff" : (entry.isPositive ? "var(--success)" : "var(--danger)")} 
                      fillOpacity={entry.id === selectedPoint?.index ? 1 : 0.6}
                      r={entry.id === selectedPoint?.index ? 6 : 4}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Dynamic Side Panel */}
        {selectedPoint && (
           <div style={{ flex: 1, overflowY: "auto", paddingRight: "5px" }}>
             <SidePanel 
               sessionData={sessionData} 
               analysisResult={analysisResult} 
               selectedData={selectedPoint} 
               onClose={() => setSelectedPoint(null)} 
             />
           </div>
        )}
      </div>

      <div className="nav-buttons">
        <button className="btn-secondary" onClick={onPrev}><ArrowLeft size={18} /> Back</button>
        <button className="btn-primary" onClick={onNext}>Initiate Mitigation <ArrowRight size={18} /></button>
      </div>
    </div>
  );
}
