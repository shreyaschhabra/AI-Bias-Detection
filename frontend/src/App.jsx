import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ShieldAlert } from "lucide-react";

import UploadStep from "./components/Wizard/UploadStep";
import DiscoveryStep from "./components/Wizard/DiscoveryStep";
import GlobalMetricsStep from "./components/Wizard/GlobalMetricsStep";
import LocalExplorerStep from "./components/Wizard/LocalExplorerStep";
import MitigationStep from "./components/Wizard/MitigationStep";

import "./App.css";

const STEPS = {
  UPLOAD: 0,
  DISCOVERY: 1,
  METRICS: 2,
  EXPLORER: 3,
  MITIGATION: 4,
};

function App() {
  const [currentStep, setCurrentStep] = useState(STEPS.UPLOAD);
  
  // Track our data globally as we move through the wizard
  const [sessionData, setSessionData] = useState({
    file: null,
    model: null,
    target_column: "",
    sensitive_column: "",
    privileged_value: "",
    positive_label: "1"
  });
  
  const [analysisResult, setAnalysisResult] = useState(null);

  const nextStep = () => setCurrentStep((prev) => prev + 1);
  const prevStep = () => setCurrentStep((prev) => Math.max(0, prev - 1));

  const variants = {
    initial: { opacity: 0, x: 20 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -20 },
  };

  return (
    <div className="layout">
      <nav className="navbar glass-panel">
        <div className="logo">
          <ShieldAlert className="icon-accent" /> AutoBias
        </div>
        <div className="steps-indicator">
          {["Data & Model", "Discovery", "Fairness Metrics", "Local Explorer", "Auto-Mitigate"].map((label, idx) => (
            <div 
              key={label} 
              className={`step-dot ${idx === currentStep ? "active" : ""} ${idx < currentStep ? "done" : ""}`}
            >
              {label}
            </div>
          ))}
        </div>
      </nav>

      <main className="wizard-container">
        <AnimatePresence mode="wait">
          {currentStep === STEPS.UPLOAD && (
            <motion.div key="step-0" variants={variants} initial="initial" animate="animate" exit="exit" className="step-wrapper">
              <UploadStep 
                sessionData={sessionData} 
                setSessionData={setSessionData} 
                onNext={nextStep} 
                setAnalysisResult={setAnalysisResult} 
              />
            </motion.div>
          )}
          {currentStep === STEPS.DISCOVERY && (
            <motion.div key="step-1" variants={variants} initial="initial" animate="animate" exit="exit" className="step-wrapper">
              <DiscoveryStep 
                analysisResult={analysisResult} 
                onNext={nextStep} 
                onPrev={prevStep} 
              />
            </motion.div>
          )}
          {currentStep === STEPS.METRICS && (
            <motion.div key="step-2" variants={variants} initial="initial" animate="animate" exit="exit" className="step-wrapper">
              <GlobalMetricsStep 
                analysisResult={analysisResult} 
                onNext={nextStep} 
                onPrev={prevStep} 
              />
            </motion.div>
          )}
          {currentStep === STEPS.EXPLORER && (
            <motion.div key="step-3" variants={variants} initial="initial" animate="animate" exit="exit" className="step-wrapper">
              <LocalExplorerStep 
                sessionData={sessionData} 
                analysisResult={analysisResult} 
                onNext={nextStep} 
                onPrev={prevStep} 
              />
            </motion.div>
          )}
          {currentStep === STEPS.MITIGATION && (
            <motion.div key="step-4" variants={variants} initial="initial" animate="animate" exit="exit" className="step-wrapper">
              <MitigationStep 
                sessionData={sessionData} 
                analysisResult={analysisResult} 
                onPrev={prevStep} 
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
