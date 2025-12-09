import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { AlertCircle, CheckCircle } from 'lucide-react';

const DischargePrediction = ({ data, predictionData }) => {
  if (!data) return <div className="card">Loading discharge data...</div>;

  const chartData = data.labels.map((label, index) => ({
    time: label,
    discharge: data.data[index],
    floodWarning: data.floodWarning,
    floodWatch: data.floodWatch,
    extremeLevel: data.extremeLevel
  }));

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'extreme': return 'bg-red-600';
      case 'danger': return 'bg-orange-500';
      case 'warning': return 'bg-yellow-400';
      default: return 'bg-green-500';
    }
  };

  const getSeverityLabel = (severity) => {
    switch(severity) {
      case 'extreme': return 'Extreme';
      case 'danger': return 'Danger';
      case 'warning': return 'Warning';
      default: return 'Normal';
    }
  };

  return (
    <div className="card compact-card">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Predicted Runoff & Return Period Comparison</h2>
          <p className="text-xs text-gray-600 mt-0.5">7-Day Discharge Forecast with Return Period Thresholds</p>
        </div>
        {data.maxSeverity && (
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-white text-xs font-semibold ${getSeverityColor(data.maxSeverity)}`}>
            {data.maxSeverity !== 'normal' ? <AlertCircle size={16} /> : <CheckCircle size={16} />}
            {getSeverityLabel(data.maxSeverity)}
          </div>
        )}
      </div>
      
      {predictionData && predictionData.basin && (
        <div className="mb-3 p-2 bg-gray-50 rounded-lg text-xs">
          <p className="text-gray-600">
            <span className="font-medium">Gauge:</span> {predictionData.basin.gauge_id}
            {predictionData.basin.distance_km && (
              <span className="ml-3">
                <span className="font-medium">Distance:</span> {predictionData.basin.distance_km.toFixed(2)} km
              </span>
            )}
          </p>
        </div>
      )}

      <div className="h-64 mb-3">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="time" 
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <YAxis 
              label={{ value: 'Discharge (m³/s)', angle: -90, position: 'insideLeft' }}
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <Tooltip 
              formatter={(value, name) => {
                if (name === 'discharge') return [`${Number(value).toFixed(2)} m³/s`, 'Predicted Discharge'];
                if (name === 'floodWarning') return [`${Number(value).toFixed(2)} m³/s`, 'Warning Level (2-year)'];
                if (name === 'floodWatch') return [`${Number(value).toFixed(2)} m³/s`, 'Danger Level (5-year)'];
                if (name === 'extremeLevel') return [`${Number(value).toFixed(2)} m³/s`, 'Extreme Level (20-year)'];
                return value;
              }}
              contentStyle={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                padding: '10px'
              }}
            />
            <Legend 
              wrapperStyle={{ paddingTop: '20px' }}
              iconType="line"
            />
            <Line 
              type="monotone" 
              dataKey="discharge" 
              name="Predicted Discharge"
              stroke="#3b82f6" 
              strokeWidth={3}
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 5 }}
              activeDot={{ r: 7 }}
            />
            <Line 
              type="monotone" 
              dataKey="floodWarning" 
              name="Warning Level (2-year)"
              stroke="#eab308" 
              strokeDasharray="8 4"
              strokeWidth={2}
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="floodWatch" 
              name="Danger Level (5-year)"
              stroke="#f97316" 
              strokeDasharray="8 4"
              strokeWidth={2}
              dot={false}
            />
            {data.extremeLevel && (
              <Line 
                type="monotone" 
                dataKey="extremeLevel" 
                name="Extreme Level (20-year)"
                stroke="#dc2626" 
                strokeDasharray="8 4"
                strokeWidth={2}
                dot={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      {predictionData && predictionData.thresholds && (
        <div className="mt-3 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
          <h3 className="text-xs font-semibold mb-2 text-gray-800">Return Period Thresholds Reference:</h3>
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-2 rounded">
              <div className="font-semibold text-yellow-800 text-xs mb-0.5">Warning</div>
              <div className="text-sm font-bold text-yellow-900">
                {predictionData.thresholds["2"]?.toFixed(2) || 'N/A'} m³/s
              </div>
              <div className="text-xs text-yellow-700">2-year</div>
            </div>
            <div className="bg-orange-50 border-l-4 border-orange-400 p-2 rounded">
              <div className="font-semibold text-orange-800 text-xs mb-0.5">Danger</div>
              <div className="text-sm font-bold text-orange-900">
                {predictionData.thresholds["5"]?.toFixed(2) || 'N/A'} m³/s
              </div>
              <div className="text-xs text-orange-700">5-year</div>
            </div>
            <div className="bg-red-50 border-l-4 border-red-400 p-2 rounded">
              <div className="font-semibold text-red-800 text-xs mb-0.5">Extreme</div>
              <div className="text-sm font-bold text-red-900">
                {predictionData.thresholds["20"]?.toFixed(2) || 'N/A'} m³/s
              </div>
              <div className="text-xs text-red-700">20-year</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DischargePrediction;
