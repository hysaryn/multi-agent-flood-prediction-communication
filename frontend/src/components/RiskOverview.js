import React from 'react';

const RiskOverview = ({ riskLevel }) => {
  const getRiskColor = (level) => {
    if (level.toLowerCase().includes('high')) return 'high';
    if (level.toLowerCase().includes('moderate')) return 'moderate';
    return 'low';
  };

  const getRecommendations = (level) => {
    if (level.toLowerCase().includes('high')) {
      return "Evacuate immediately if in flood-prone areas. Avoid all travel unless absolutely necessary.";
    }
    if (level.toLowerCase().includes('moderate')) {
      return "Prepare sandbags and emergency supply kit. Monitor local alerts.";
    }
    return "Stay informed about weather updates. Prepare basic emergency kit.";
  };

  return (
    <div className="card">
      <h2 className="text-xl font-semibold mb-4">Risk Overview</h2>
      <div className={`risk-card ${getRiskColor(riskLevel)}`}>
        <div className="text-lg font-bold mb-2">{riskLevel}</div>
        <div className="text-sm opacity-90 mb-4">{getRecommendations(riskLevel)}</div>
        <button className="bg-white bg-opacity-20 hover:bg-opacity-30 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
          View Recommendations
        </button>
      </div>
    </div>
  );
};

export default RiskOverview;
