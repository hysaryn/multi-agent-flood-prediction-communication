import React, { useState, useEffect } from 'react';

const OfficialGuide = () => {
  const [content, setContent] = useState('');
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadOfficialGuide();
  }, []);

  const loadOfficialGuide = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/official-guide');
      const data = await response.json();
      setContent(data.content);
      setSources(data.sources);
    } catch (error) {
      console.error('Error loading official guide:', error);
      setContent('Unable to load official guide at this time.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="p-4">Loading official guide...</div>;
  }

  return (
    <div>
      <h3 className="text-lg font-semibold mb-4">Official Emergency Guidelines</h3>
      <div className="prose max-w-none">
        <div className="whitespace-pre-wrap text-sm leading-relaxed">
          {content}
        </div>
      </div>
      {sources.length > 0 && (
        <div className="mt-4 pt-4 border-t">
          <h4 className="text-sm font-medium text-gray-600 mb-2">Sources:</h4>
          <ul className="text-xs text-gray-500 space-y-1">
            {sources.map((source, index) => (
              <li key={index}>• {source}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default OfficialGuide;
