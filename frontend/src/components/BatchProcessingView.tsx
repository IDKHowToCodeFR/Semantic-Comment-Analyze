"use client";

import React, { useState } from "react";
import { Button } from "@/components/Button";
import { FeatureCard } from "@/components/FeatureCard";
import { Upload, Download } from "lucide-react";

export const BatchProcessingView = () => {
  const [file, setFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("text");
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [previewLines, setPreviewLines] = useState<string[]>([]);
  const [processedCount, setProcessedCount] = useState(0);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setDownloadUrl(null);
      setPreviewLines([]);
      setProcessedCount(0);
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setDownloadUrl(null);
    setPreviewLines([]);
    setProcessedCount(0);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("threshold", threshold.toString());
      formData.append("column", targetColumn);

      const res = await fetch("http://127.0.0.1:8000/api/batch", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(await res.text());

      const reader = res.body?.getReader();
      if (!reader) throw new Error("Stream not available");

      const decoder = new TextDecoder("utf-8");
      let csvStr = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        csvStr += decoder.decode(value, { stream: true });
        const lines = csvStr.split("\n").filter((l) => l.trim().length > 0);
        
        // Header doesn't count towards processed rows
        setProcessedCount(Math.max(0, lines.length - 1));
        setPreviewLines(lines.slice(0, 6)); // Keep preview fast
      }
      
      csvStr += decoder.decode();

      // Create download URL after complete
      const blob = new Blob([csvStr], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Batch processing failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="py-24 px-8 max-w-[1200px] mx-auto flex flex-col gap-12">
      <div>
        <div className="flex items-center justify-between pb-2">
          <h2 className="font-serif text-[36px] font-[300] tracking-[-0.36px] text-ink">
            Batch Processing
          </h2>
          <span className="bg-surface-strong text-ink text-[12px] font-semibold tracking-[0.96px] uppercase rounded-full px-[10px] py-[4px]">
            Sarcasm Engine Active
          </span>
        </div>
        <p className="font-sans text-[16px] text-body tracking-[0.16px]">
          Upload a CSV file to process multiple texts simultaneously.
        </p>
      </div>

      <FeatureCard className="flex flex-col gap-6 p-6 rounded-[16px]">
        {/* Upload Area */}
        <div className="border border-hairline-strong rounded-[12px] p-6 flex flex-col items-center justify-center bg-canvas-soft relative hover:border-ink transition-colors cursor-pointer">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
          />
          <Upload className="text-muted mb-3" size={24} />
          {file ? (
            <p className="text-ink font-medium font-sans text-[16px]">{file.name}</p>
          ) : (
            <p className="text-muted font-sans text-[15px]">
              Drag and drop a CSV file, or click to select
            </p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="flex flex-col gap-2">
            <label className="text-[14px] font-sans font-medium text-body-strong">Target Column</label>
            <input
              type="text"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full bg-surface-card rounded-[8px] px-4 py-3 text-ink font-sans text-[16px] border border-hairline-strong focus:border-ink focus:border-[2px] focus:outline-none transition-all"
              placeholder="e.g. text"
            />
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-body-strong font-sans font-medium text-[14px]">
              <label>Threshold</label>
              <span>{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-ink mt-3"
            />
          </div>
        </div>

        <div className="flex items-center gap-4 mt-2">
          <Button onClick={handleProcess} disabled={!file || loading} className="w-full md:w-auto">
            {loading ? "Processing..." : "Process CSV"}
          </Button>
          {loading && (
            <p className="text-[14px] font-sans text-muted animate-pulse">
              Streaming... processed {processedCount} rows
            </p>
          )}
        </div>
        
        {error && <p className="text-semantic-error text-[14px] font-sans mt-2">{error}</p>}
      </FeatureCard>

      {/* Results Preview */}
      {previewLines.length > 0 && (
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between border-b border-hairline pb-4">
            <h3 className="font-serif text-[24px] font-[300] text-ink">Live Preview</h3>
            {downloadUrl && (
              <a
                href={downloadUrl}
                download={`processed_${file?.name || "data.csv"}`}
                className="inline-flex items-center gap-2 text-[15px] font-sans font-medium text-ink hover:opacity-70 transition-opacity"
              >
                <Download size={16} />
                Download Full CSV
              </a>
            )}
          </div>

          <div className="overflow-x-auto border border-hairline rounded-[12px]">
            <table className="w-full text-left text-[14px] font-sans whitespace-nowrap">
              <thead>
                <tr className="bg-surface-strong border-b border-hairline text-body-strong">
                  {previewLines[0].split(",").map((header, i) => (
                    <th key={i} className="px-4 py-3 font-medium">
                      {header.trim()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-surface-card divide-y divide-hairline text-ink">
                {previewLines.slice(1).map((row, i) => (
                  <tr key={i}>
                    {row.split(",").map((cell, j) => (
                      <td key={j} className="px-4 py-3">
                        {cell.trim().length > 40 ? cell.trim().substring(0, 40) + "..." : cell.trim()}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
