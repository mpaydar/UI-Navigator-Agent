"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const [goal, setGoal] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!goal.trim()) {
      setError("Please enter a goal");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const response = await fetch("http://localhost:8000/api/task", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ goal }),
      });

      if (!response.ok) {
        throw new Error("Failed to submit task");
      }

      const data = await response.json();
      
      // Redirect to snapshots page
      router.push(`/snapshots/${data.session_id}`);
    } catch (err) {
      setError("Failed to submit task. Make sure the backend is running.");
      console.error("Error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen">
      {/* Background Image Layer with Opacity */}
      <div 
        className="absolute inset-0 opacity-20 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: 'url(https://techvidvan.com/tutorials/wp-content/uploads/2025/03/ai-agents.webp)'
        }}
      />
      
      {/* Content Layer */}
      <div className="relative z-10 flex flex-col items-center pt-8">
        <h1 className="text-2xl font-bold text-center">Web Navigator Agent</h1>
        
        <div className="flex flex-col items-center justify-center mt-20">
          <form onSubmit={handleSubmit} className="w-full max-w-md">
            <input
              type="text"
              placeholder="Enter your goal"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              disabled={isLoading}
              className="w-full p-3 rounded-md border border-gray-300 bg-white text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            />
            
            {error && (
              <p className="text-red-500 text-sm mt-2 text-center">{error}</p>
            )}
            
            <div className="flex flex-col items-center justify-center">
              <button 
                type="submit" 
                disabled={isLoading}
                className="bg-blue-500 text-white px-6 py-3 rounded-md mt-4 hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Processing...
                  </>
                ) : (
                  "Start Agent"
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
