import React, { useState, useEffect } from 'react';
import './App.css';
import Header from './components/Header';
import RiskOverview from './components/RiskOverview';
import DischargePrediction from './components/DischargePrediction';
import InformationTabs from './components/InformationTabs';
import Chatbot from './components/Chatbot';

function App() {
  const [activeTab, setActiveTab] = useState('OfficialGuide');
  const [riskLevel, setRiskLevel] = useState('Moderate to High Over 7 Days');
  const [dischargeData, setDischargeData] = useState(null);

  useEffect(() => {
    // Load initial data
    loadRiskData();
    loadDischargeData();
  }, []);

  const loadRiskData = async () => {
    try {
      const response = await fetch('http://localhost:8000/live?location=Hope, BC');
      const data = await response.json();
      setRiskLevel(data.flood_risk);
    } catch (error) {
      console.error('Error loading risk data:', error);
    }
  };

  const loadDischargeData = async () => {
    // Mock data for now
    const mockData = {
      labels: ['24h', '48h', '72h', '96h', '120h', '144h', '168h'],
      data: [150, 180, 220, 350, 450, 600, 750],
      floodWarning: 200,
      floodWatch: 650
    };
    setDischargeData(mockData);
  };

  return (
    <div className="App">
      <Header />
      <div className="main-content">
        <RiskOverview riskLevel={riskLevel} />
        <DischargePrediction data={dischargeData} />
        <InformationTabs 
          activeTab={activeTab} 
          setActiveTab={setActiveTab}
        />
        <Chatbot />
      </div>
    </div>
  );
}

export default App;
