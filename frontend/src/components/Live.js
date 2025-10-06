import React, { useState, useEffect } from 'react';

const Live = () => {
  const [weatherData, setWeatherData] = useState(null);
  const [floodRisk, setFloodRisk] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadLiveData();
    const interval = setInterval(loadLiveData, 300000); // Update every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const loadLiveData = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/live?location=Hope, BC');
      const data = await response.json();
      setWeatherData(data.weather_data);
      setFloodRisk(data.flood_risk);
      setRecommendations(data.recommendations);
    } catch (error) {
      console.error('Error loading live data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !weatherData) {
    return <div className="p-4">Loading live data...</div>;
  }

  return (
    <div>
      <h3 className="text-lg font-semibold mb-4">Live Weather Data</h3>
      
      {weatherData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-sm text-gray-600">Temperature</div>
            <div className="text-xl font-bold">{weatherData.temperature}°C</div>
          </div>
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-sm text-gray-600">Precipitation</div>
            <div className="text-xl font-bold">{weatherData.precipitation}mm</div>
          </div>
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-sm text-gray-600">Humidity</div>
            <div className="text-xl font-bold">{weatherData.humidity}%</div>
          </div>
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-sm text-gray-600">Wind Speed</div>
            <div className="text-xl font-bold">{weatherData.wind_speed} km/h</div>
          </div>
        </div>
      )}

      <div className="mb-6">
        <h4 className="text-md font-medium mb-2">Current Flood Risk: 
          <span className={`ml-2 px-2 py-1 rounded text-sm ${
            floodRisk === 'High' ? 'bg-red-100 text-red-800' :
            floodRisk === 'Moderate' ? 'bg-yellow-100 text-yellow-800' :
            'bg-green-100 text-green-800'
          }`}>
            {floodRisk}
          </span>
        </h4>
      </div>

      {recommendations.length > 0 && (
        <div>
          <h4 className="text-md font-medium mb-2">Current Recommendations:</h4>
          <ul className="space-y-1">
            {recommendations.map((rec, index) => (
              <li key={index} className="flex items-start">
                <span className="text-blue-500 mr-2">•</span>
                <span className="text-sm">{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-4 text-xs text-gray-500">
        Last updated: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

export default Live;
