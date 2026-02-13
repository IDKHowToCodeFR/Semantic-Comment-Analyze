"use client";

import React, { useState } from "react";
import { Button } from "@/components/Button";
import { FeatureCard } from "@/components/FeatureCard";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";

type ExplainabilityData = { word: string; contribution: number; primary_intent: string }[];

type AnalysisResult = {
  intent: { labels: string[]; scores: number[]; all_intents: Record<string, number> };
  sentiment: { label: string; score: number };
  explainability: ExplainabilityData;
  tone?: string;
  urgency?: string;
  recommended_action?: string;
};

const INTENT_COLORS: Record<string, string> = {
  "Bug Report": "239, 68, 68", // #ef4444 (Red)
  "Complaint": "249, 115, 22", // #f97316 (Orange)
  "Feature Request": "139, 92, 246", // #8b5cf6 (Purple)
  "Praise": "34, 197, 94", // #22c55e (Green)
  "Question": "59, 130, 246", // #3b82f6 (Blue)
};

const WordContributionHeatmap = ({ data }: { data: ExplainabilityData }) => {
  if (!data || data.length === 0) return null;
  
  // Find max absolute contribution for scaling
  const maxAbs = Math.max(...data.map(d => Math.abs(d.contribution)), 0.001);

  return (
    <div className="flex flex-col gap-4">
      <p className="text-[13px] text-muted mb-1 font-medium">
        Highlighting word contributions mapped to their primary intents.
      </p>
      <div className="leading-relaxed whitespace-pre-wrap text-[16px] p-5 bg-canvas-soft rounded-lg border border-hairline font-sans text-ink flex flex-wrap gap-x-1 gap-y-1.5 shadow-inner">
        {data.map((item, idx) => {
          const alpha = Math.min(Math.abs(item.contribution) / maxAbs, 1);
          const colorBase = INTENT_COLORS[item.primary_intent];
          const bgColor = colorBase && item.contribution > 0 ? `rgba(${colorBase}, ${alpha * 0.6})` : 'transparent';
          
          return (
            <span
              key={idx}
              style={{ backgroundColor: alpha > 0.05 ? bgColor : 'transparent' }}
              className={`px-1 rounded-sm transition-colors cursor-default ${alpha > 0.3 ? 'font-medium' : ''}`}
              title={`Tilts toward ${item.primary_intent}: +${(item.contribution * 100).toFixed(1)}%`}
            >
              {item.word}
            </span>
          );
        })}
      </div>
      
      {/* Legend */}
      <div className="flex flex-wrap gap-5 mt-2 pt-4 border-t border-hairline">
        {Object.entries(INTENT_COLORS).map(([intent, rgb]) => (
          <div key={intent} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: `rgb(${rgb})` }} />
            <span className="text-[12px] text-muted-strong font-medium">{intent}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const INTENTS = ["Bug Report", "Complaint", "Feature Request", "Praise", "Question"];

export const SingleAnalysisView = () => {
  const [text, setText] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, threshold, include_explanation: true }),
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to connect to API.");
    } finally {
      setLoading(false);
    }
  };

  const getSentimentData = () => {
    if (!result) return [];
    const { label, score } = result.sentiment;
    const pos = label === "POSITIVE" ? score : 0;
    const neg = label === "NEGATIVE" ? score : 0;
    const neu = label === "NEUTRAL" ? score : 0;
    const sarc = label === "NEGATIVE (SARCASTIC)" ? score : 0;
    return [
      { name: "Positive", value: pos, color: "#16a34a" },
      { name: "Neutral", value: neu, color: "#a8a29e" },
      { name: "Negative", value: neg, color: "#dc2626" },
      { name: "Sarcastic", value: sarc, color: "#8b5cf6" },
    ];
  };

  const getRadarData = () => {
    if (!result || !result.intent.all_intents) return [];
    return INTENTS.map(intent => ({
      subject: intent,
      A: (result.intent.all_intents[intent] || 0) * 100, // Scale to 100
      fullMark: 100,
    }));
  };

  return (
    <div className="p-8 max-w-4xl mx-auto flex flex-col gap-12">
      {/* Input Section (Top) */}
      <div className="flex flex-col gap-6">
        <div>
          <div className="flex items-center justify-between border-b border-hairline pb-2 mb-4">
            <h2 className="font-serif text-[32px] text-ink">
              Analysis Input
            </h2>
            <span className="bg-surface-strong text-ink text-[12px] font-semibold tracking-[0.96px] uppercase rounded-full px-[10px] py-[4px]">
              Sarcasm Engine Active
            </span>
          </div>
          <textarea
            className="w-full h-32 bg-surface-card rounded-md p-4 text-ink font-sans text-[15px] border border-hairline-strong focus:border-ink focus:outline-none focus:ring-1 focus:ring-ink transition-all resize-none shadow-sm"
            placeholder="Paste text or customer feedback here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>

        <div className="flex flex-col md:flex-row gap-6 items-end">
          <div className="flex-1 flex flex-col gap-2 w-full">
            <div className="flex justify-between items-center text-body-strong font-medium text-[14px]">
              <span>Confidence Threshold</span>
              <span>{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-ink"
            />
          </div>

          <div className="w-full md:w-auto">
            <Button onClick={handleAnalyze} disabled={loading || !text.trim()} className="w-full md:w-48 shadow-sm">
              {loading ? "Analyzing..." : "Execute Analysis"}
            </Button>
          </div>
        </div>
        
        {error && <p className="text-semantic-error text-[14px]">{error}</p>}
      </div>

      {/* Output Section (Bottom) */}
      <div className="flex flex-col gap-6">
        <h2 className="font-serif text-[32px] text-ink border-b border-hairline pb-2">
          Results
        </h2>

        {result ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col gap-6"
          >
            {/* Top Row: Intent Radar & Sentiment */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <FeatureCard 
                title="Intent Distribution" 
                className="p-4"
              >
                <div className="h-64 w-full mt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={getRadarData()}>
                      <PolarGrid stroke="#e7e5e4" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: "#777169", fontSize: 11 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                      <Radar
                        name="Intent Confidence"
                        dataKey="A"
                        stroke="#292524"
                        fill="#292524"
                        fillOpacity={0.2}
                      />
                      <Tooltip contentStyle={{ fontSize: "12px", borderRadius: "4px" }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </FeatureCard>

              <FeatureCard 
                title="Sentiment" 
                className="p-4"
              >
                <div className="h-64 w-full mt-2 flex items-center justify-center">
                  <ResponsiveContainer width="100%" height="80%">
                    <BarChart data={getSentimentData()} margin={{ top: 20, right: 0, left: 0, bottom: 0 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 13, fill: "#777169", fontWeight: 500 }} axisLine={false} tickLine={false} />
                      <Tooltip cursor={{ fill: "transparent" }} contentStyle={{ fontSize: "12px", borderRadius: "4px" }} />
                      <Bar dataKey="value" radius={[4, 4, 4, 4]} maxBarSize={40}>
                        {getSentimentData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </FeatureCard>
            </div>

            {/* Middle Row: Tone, Urgency, Action */}
            {(result.tone || result.urgency) && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Left Side: Stacked Tone and Urgency */}
                <div className="flex flex-col gap-6">
                  <FeatureCard 
                    title="Detected Tone" 
                    className="p-4 flex-1"
                  >
                    <div className="flex-1 flex items-center justify-center min-h-[40px] text-[36px] font-serif font-light text-ink pb-2">
                      {result.tone}
                    </div>
                  </FeatureCard>
                  
                  <FeatureCard 
                    title="Urgency Level" 
                    className="p-4 flex-1"
                  >
                    <div className="flex-1 flex items-center justify-center min-h-[40px] text-[24px] font-sans font-bold pb-2" style={{ color: result.urgency === 'High' ? '#dc2626' : result.urgency === 'Medium' ? '#f97316' : '#16a34a' }}>
                      {result.urgency}
                    </div>
                  </FeatureCard>
                </div>

                {/* Right Side: Recommended Action (Full height) */}
                <FeatureCard 
                  title="Recommended Action" 
                  className="p-4 h-full"
                >
                  <div className="flex-1 flex items-center justify-center min-h-[160px] text-[48px] font-serif font-light text-ink text-center px-4 leading-[1.08] tracking-tight pb-4">
                    {result.recommended_action}
                  </div>
                </FeatureCard>
                
              </div>
            )}

            {/* Bottom Row: Explainability */}
            <FeatureCard title="Word Contribution (Explainability)" className="p-4">
              <div className="mt-2">
                <WordContributionHeatmap data={result.explainability} />
              </div>
            </FeatureCard>
          </motion.div>
        ) : (
          <div className="h-48 flex items-center justify-center border border-dashed border-hairline-strong rounded-lg p-8 text-muted text-center text-[14px]">
            Run an analysis to view structure, intent, and sentiment results.
          </div>
        )}
      </div>
    </div>
  );
};
