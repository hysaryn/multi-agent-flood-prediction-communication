import React, { useState, useEffect } from 'react';
import { X, Send } from 'lucide-react';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Add welcome message
    setMessages([
      {
        id: 1,
        text: "Hello! I'm your flood prediction assistant. I can help you with emergency guidelines, current conditions, and social media insights. How can I help you today?",
        sender: 'bot',
        timestamp: new Date()
      }
    ]);
  }, []);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          location: 'Hope, BC'
        }),
      });

      const data = await response.json();
      
      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        timestamp: new Date(),
        sources: data.sources,
        confidence: data.confidence
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: "I'm sorry, I'm having trouble connecting right now. Please try again later.",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 bg-blue-500 text-white p-4 rounded-full shadow-lg hover:bg-blue-600 transition-colors"
      >
        💬
      </button>
    );
  }

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h3 className="font-semibold">Chatbot</h3>
        <button
          onClick={() => setIsOpen(false)}
          className="text-gray-500 hover:text-gray-700"
        >
          <X size={20} />
        </button>
      </div>
      
      <div className="chatbot-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`mb-3 ${
              message.sender === 'user' ? 'text-right' : 'text-left'
            }`}
          >
            <div
              className={`inline-block max-w-xs p-3 rounded-lg text-sm ${
                message.sender === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-800'
              }`}
            >
              {message.text}
              {message.sources && (
                <div className="mt-2 text-xs opacity-75">
                  Sources: {message.sources.join(', ')}
                </div>
              )}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="text-left">
            <div className="inline-block bg-gray-200 p-3 rounded-lg text-sm">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="chatbot-input">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about flood conditions..."
          className="flex-1 p-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || !inputMessage.trim()}
          className="px-4 py-2 bg-blue-500 text-white rounded-r-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
