import React, { useState } from 'react';
import { MapPin, Search, X } from 'lucide-react';

const LocationInput = ({ onLocationChange, defaultLocation = "Hope, BC" }) => {
  const [location, setLocation] = useState(defaultLocation);
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmedLocation = location.trim();
    
    if (!trimmedLocation) {
      setError('Please enter a location');
      return;
    }

    setError('');
    setIsEditing(false);
    onLocationChange(trimmedLocation);
  };

  const handleCancel = () => {
    setLocation(defaultLocation);
    setError('');
    setIsEditing(false);
  };

  if (!isEditing) {
    return (
      <div className="flex items-center gap-3 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-200 shadow-sm">
        <div className="bg-blue-500 p-2 rounded-lg">
          <MapPin className="text-white" size={22} />
        </div>
        <div className="flex-1">
          <p className="text-xs text-gray-600 uppercase tracking-wide mb-1">Current Location</p>
          <p className="text-lg font-bold text-gray-900">{location}</p>
        </div>
        <button
          onClick={() => setIsEditing(true)}
          className="px-4 py-2 text-sm font-medium bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all shadow-md hover:shadow-lg"
        >
          Change Location
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-200 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="bg-blue-500 p-2 rounded-lg">
          <MapPin className="text-white" size={22} />
        </div>
        <div className="flex-1">
          <input
            type="text"
            value={location}
            onChange={(e) => {
              setLocation(e.target.value);
              setError('');
            }}
            placeholder="Enter address, city name, or coordinates (e.g., Surrey, BC)"
            className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base"
            autoFocus
          />
          {error && (
            <p className="mt-2 text-sm text-red-600 font-medium">{error}</p>
          )}
          <p className="mt-2 text-xs text-gray-600">
            💡 Enter a city name, full address, or coordinates (latitude, longitude)
          </p>
        </div>
        <button
          type="submit"
          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all shadow-md hover:shadow-lg flex items-center gap-2 font-medium"
        >
          <Search size={18} />
          Search
        </button>
        <button
          type="button"
          onClick={handleCancel}
          className="px-4 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
        >
          <X size={18} />
        </button>
      </div>
    </form>
  );
};

export default LocationInput;


