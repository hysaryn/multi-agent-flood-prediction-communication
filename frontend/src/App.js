import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import Header from './components/Header';
import DischargePrediction from './components/DischargePrediction';
import SeverityAssessment from './components/SeverityAssessment';
import ActionPlan from './components/ActionPlan';
import LocationInput from './components/LocationInput';

function App() {
  const [dischargeData, setDischargeData] = useState(null);
  const [location, setLocation] = useState('Vancouver, BC');
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadDischargeData = useCallback(async (loc = location) => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/predict?q=${encodeURIComponent(loc)}`);
      const data = await response.json();
      
      if (data.ok && data.glofas && data.glofas.forecast) {
        // Transform API data to chart format
        const forecast = data.glofas.forecast;
        const labels = forecast.map((point, index) => {
          const date = new Date(point.date);
          return `${date.getDate()}/${date.getMonth() + 1}`;
        });
        const dischargeValues = forecast.map(point => point.discharge);
        
        // Get thresholds
        const warningThreshold = data.thresholds?.["2"] || 0;
        const dangerThreshold = data.thresholds?.["5"] || 0;
        const extremeThreshold = data.thresholds?.["20"] || 0;
        
        setDischargeData({
          labels: labels,
          data: dischargeValues,
          floodWarning: warningThreshold,
          floodWatch: dangerThreshold,
          extremeLevel: extremeThreshold,
          current: data.glofas.current,
          maxSeverity: data.max_severity || 'normal'
        });
        
        setPredictionData(data);
      } else {
        console.error('Invalid prediction data:', data);
      }
    } catch (error) {
      console.error('Error loading discharge data:', error);
    } finally {
      setLoading(false);
    }
  }, [location]);

  useEffect(() => {
    // Load initial data
    loadDischargeData();
    // Note: loadDischargeData already sets predictionData
  }, [loadDischargeData]);

  const handleLocationChange = (newLocation) => {
    setLocation(newLocation);
    loadDischargeData(newLocation);
    // Note: loadDischargeData already sets predictionData
  };

  return (
    <div className="App">
      <Header />
      <div className="main-content">
        {/* Location Input - Full Width */}
        <div>
          <LocationInput 
            onLocationChange={handleLocationChange}
            defaultLocation={location}
          />
        </div>
        
        {loading && (
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-blue-600 text-sm">Loading prediction data for {location}...</p>
          </div>
        )}
        
        {/* Two Column Layout */}
        <div className="two-column-layout">
          {/* Left Column: Severity Assessment + Chart */}
          <div className="left-column">
            <SeverityAssessment 
              predictionData={predictionData} 
              maxSeverity={predictionData?.max_severity}
            />
            <DischargePrediction 
              data={dischargeData} 
              predictionData={predictionData} 
            />
          </div>
          
          {/* Right Column: Action Plan */}
          <div className="right-column">
            <ActionPlan />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
