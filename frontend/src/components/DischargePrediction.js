import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const DischargePrediction = ({ data }) => {
  if (!data) return <div className="card">Loading discharge data...</div>;

  const chartData = data.labels.map((label, index) => ({
    time: label,
    discharge: data.data[index],
    floodWarning: data.floodWarning,
    floodWatch: data.floodWatch
  }));

  return (
    <div className="card">
      <h2 className="text-xl font-semibold mb-4">Discharge Prediction</h2>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis label={{ value: 'Discharge (m³/s)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Line 
              type="monotone" 
              dataKey="discharge" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="floodWarning" 
              stroke="#ef4444" 
              strokeDasharray="5 5"
              strokeWidth={2}
            />
            <Line 
              type="monotone" 
              dataKey="floodWatch" 
              stroke="#f59e0b" 
              strokeDasharray="5 5"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-2 mt-4">
        {['24 h', '48 h', '72 h', 'High', '95 h', '168 h'].map((period) => (
          <button
            key={period}
            className={`px-3 py-1 rounded text-sm ${
              period === 'High' 
                ? 'bg-orange-500 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {period}
          </button>
        ))}
      </div>
    </div>
  );
};

export default DischargePrediction;
