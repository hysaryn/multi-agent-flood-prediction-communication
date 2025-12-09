import React from 'react';
import { AlertCircle, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

const SeverityAssessment = ({ predictionData, maxSeverity }) => {
  if (!predictionData) {
    return (
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Severity Assessment</h2>
        <p className="text-gray-500">Loading assessment data...</p>
      </div>
    );
  }

  const severity = maxSeverity || predictionData.max_severity || 'normal';
  const thresholds = predictionData.thresholds || {};
  const currentDischarge = predictionData.glofas?.current || 0;
  const forecast = predictionData.glofas?.forecast || [];

  const severityConfig = {
    normal: {
      label: 'Normal',
      color: 'bg-green-500',
      bgColor: 'bg-green-50',
      textColor: 'text-green-800',
      borderColor: 'border-green-200',
      icon: CheckCircle,
      description: 'Discharge levels are within normal ranges. No immediate flood risk.',
      recommendation: 'Continue monitoring conditions. Stay informed about weather updates.'
    },
    warning: {
      label: 'Warning',
      color: 'bg-yellow-500',
      bgColor: 'bg-yellow-50',
      textColor: 'text-yellow-800',
      borderColor: 'border-yellow-200',
      icon: AlertTriangle,
      description: 'Discharge approaching 2-year return period threshold. Increased flood risk possible.',
      recommendation: 'Prepare emergency supplies. Monitor local alerts closely. Review evacuation routes.'
    },
    danger: {
      label: 'Danger',
      color: 'bg-orange-500',
      bgColor: 'bg-orange-50',
      textColor: 'text-orange-800',
      borderColor: 'border-orange-200',
      icon: AlertCircle,
      description: 'Discharge exceeds 5-year return period threshold. Significant flood risk.',
      recommendation: 'Prepare to evacuate if in flood-prone areas. Secure important documents and valuables.'
    },
    extreme: {
      label: 'Extreme',
      color: 'bg-red-600',
      bgColor: 'bg-red-50',
      textColor: 'text-red-800',
      borderColor: 'border-red-200',
      icon: XCircle,
      description: 'Discharge exceeds 20-year return period threshold. Extreme flood risk.',
      recommendation: 'Evacuate immediately if in flood-prone areas. Follow emergency services instructions.'
    }
  };

  const config = severityConfig[severity] || severityConfig.normal;
  const Icon = config.icon;

  // Find the highest discharge in forecast
  const maxDischarge = Math.max(...forecast.map(f => f.discharge || 0), currentDischarge);

  return (
    <div className="card compact-card">
      <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
        <Icon className={`w-5 h-5 ${config.textColor}`} />
        Maximum Severity Assessment
      </h2>

      {/* Main Severity Card */}
      <div className={`${config.bgColor} ${config.borderColor} border-2 rounded-lg p-4 mb-4`}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`${config.color} w-12 h-12 rounded-full flex items-center justify-center text-white`}>
              <Icon className="w-6 h-6" />
            </div>
            <div>
              <h3 className={`text-2xl font-bold ${config.textColor}`}>
                {config.label.toUpperCase()}
              </h3>
              <p className={`text-xs ${config.textColor} opacity-75`}>
                Current Maximum Severity Level
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className={`text-xl font-bold ${config.textColor}`}>
              {maxDischarge.toFixed(2)} m³/s
            </p>
            <p className={`text-xs ${config.textColor} opacity-75`}>
              Peak Discharge
            </p>
          </div>
        </div>

        <div className={`${config.borderColor} border-t pt-3 mt-3`}>
          <p className={`${config.textColor} mb-1 text-sm font-medium`}>
            {config.description}
          </p>
          <p className={`${config.textColor} text-xs`}>
            <strong>Recommendation:</strong> {config.recommendation}
          </p>
        </div>
      </div>

      {/* Current Status */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-xs text-gray-600">Current Discharge</p>
            <p className="text-lg font-bold text-gray-900">
              {currentDischarge.toFixed(2)} m³/s
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-600">Forecast Period</p>
            <p className="text-lg font-bold text-gray-900">
              {forecast.length} days
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SeverityAssessment;

