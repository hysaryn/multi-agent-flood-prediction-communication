import React, { useState, useEffect } from 'react';

const SocialMedia = () => {
  const [summary, setSummary] = useState('');
  const [sentiment, setSentiment] = useState('');
  const [keyTopics, setKeyTopics] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadSocialMediaData();
    const interval = setInterval(loadSocialMediaData, 600000); // Update every 10 minutes
    return () => clearInterval(interval);
  }, []);

  const loadSocialMediaData = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/social-media');
      const data = await response.json();
      setSummary(data.summary);
      setSentiment(data.sentiment);
      setKeyTopics(data.key_topics);
    } catch (error) {
      console.error('Error loading social media data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment.toLowerCase()) {
      case 'concerned':
        return 'text-orange-600 bg-orange-100';
      case 'positive':
        return 'text-green-600 bg-green-100';
      case 'negative':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  if (loading && !summary) {
    return <div className="p-4">Loading social media data...</div>;
  }

  return (
    <div>
      <h3 className="text-lg font-semibold mb-4">Social Media Insights</h3>
      
      {summary && (
        <div className="mb-6">
          <h4 className="text-md font-medium mb-2">Summary:</h4>
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm leading-relaxed">{summary}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <h4 className="text-md font-medium mb-2">Sentiment:</h4>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(sentiment)}`}>
            {sentiment}
          </span>
        </div>
        
        <div>
          <h4 className="text-md font-medium mb-2">Key Topics:</h4>
          <div className="flex flex-wrap gap-2">
            {keyTopics.map((topic, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
              >
                {topic}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 text-xs text-gray-500">
        Last updated: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

export default SocialMedia;
