"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Image from "next/image";

interface SessionData {
  session_id: string;
  status: string;
  goal: string;
  screenshots: string[];
  error?: string;
}

export default function SnapshotsPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;
  
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchSnapshots = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/snapshots/${sessionId}`);
        
        if (!response.ok) {
          throw new Error("Failed to fetch snapshots");
        }

        const data = await response.json();
        setSessionData(data);
        
        // Auto-refresh if still running
        if (data.status === "running") {
          setTimeout(fetchSnapshots, 3000); // Poll every 3 seconds
        }
      } catch (err) {
        setError("Failed to load snapshots. Make sure the backend is running.");
        console.error("Error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchSnapshots();
  }, [sessionId]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <svg className="animate-spin h-12 w-12 text-blue-500 mx-auto" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <p className="mt-4 text-gray-600">Loading snapshots...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-500 text-lg">{error}</p>
          <button
            onClick={() => router.push("/")}
            className="mt-4 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
          >
            Go Back Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <button
          onClick={() => router.push("/")}
          className="mb-4 text-blue-500 hover:text-blue-600 flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to Home
        </button>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h1 className="text-3xl font-bold mb-4">Agent Execution Results</h1>
          <div className="space-y-2">
            <p className="text-gray-700">
              <span className="font-semibold">Goal:</span> {sessionData?.goal}
            </p>
            <p className="text-gray-700">
              <span className="font-semibold">Status:</span>{" "}
              <span
                className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  sessionData?.status === "completed"
                    ? "bg-green-100 text-green-800"
                    : sessionData?.status === "running"
                    ? "bg-yellow-100 text-yellow-800"
                    : "bg-red-100 text-red-800"
                }`}
              >
                {sessionData?.status === "running" && (
                  <svg className="animate-spin h-4 w-4 mr-2" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                )}
                {sessionData?.status}
              </span>
            </p>
            <p className="text-gray-700">
              <span className="font-semibold">Session ID:</span> {sessionId}
            </p>
          </div>
        </div>
      </div>

      {/* Screenshots Grid */}
      <div className="max-w-7xl mx-auto">
        <h2 className="text-2xl font-bold mb-6">Step-by-Step Screenshots</h2>
        
        {sessionData?.screenshots && sessionData.screenshots.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sessionData.screenshots.map((screenshot, index) => (
              <div
                key={screenshot}
                className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-lg transition-shadow"
                onClick={() => setSelectedImage(screenshot)}
              >
                <div className="aspect-video relative bg-gray-100">
                  <img
                    src={`http://localhost:8000/api/screenshot/${screenshot}`}
                    alt={`Step ${index + 1}`}
                    className="w-full h-full object-contain"
                  />
                </div>
                <div className="p-4">
                  <p className="font-semibold text-gray-800">
                    {screenshot.includes("current") ? "Current State" : `Step ${index + 1}`}
                  </p>
                  <p className="text-sm text-gray-500">{screenshot}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-md p-8 text-center">
            <p className="text-gray-500">
              {sessionData?.status === "running"
                ? "Waiting for screenshots... The agent is still running."
                : "No screenshots available yet."}
            </p>
          </div>
        )}
      </div>

      {/* Image Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-7xl max-h-full" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute -top-12 right-0 text-white hover:text-gray-300 text-xl"
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <img
              src={`http://localhost:8000/api/screenshot/${selectedImage}`}
              alt="Full size"
              className="max-w-full max-h-[90vh] object-contain rounded-lg"
            />
          </div>
        </div>
      )}
    </div>
  );
}

