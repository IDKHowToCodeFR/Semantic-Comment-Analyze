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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setDownloadUrl(null);
      setPreviewLines([]);
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setDownloadUrl(null);
    setPreviewLines([]);

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
      const csvStr = await res.text();

      // Create download URL
      const blob = new Blob([csvStr], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);

      // Parse preview
      const lines = csvStr.split("\n").slice(0, 6); // Header + 5 rows max
      setPreviewLines(lines.filter((l) => l.trim().length > 0));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Batch processing failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto flex flex-col gap-8">
      <div>
        <h2 className="font-serif text-[28px] text-ink border-b border-hairline pb-2 mb-2">
          Batch CSV Processing
        </h2>
        <p className="text-body text-[15px]">
          Upload a CSV file to process multiple texts simultaneously.
        </p>
      </div>

      <FeatureCard className="flex flex-col gap-6">
        {/* Upload Area */}
        <div className="border-2 border-dashed border-hairline-strong rounded-lg p-8 flex flex-col items-center justify-center bg-canvas-soft relative hover:border-ink transition-colors cursor-pointer">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
          />
          <Upload className="text-muted mb-3" size={32} />
          {file ? (
            <p className="text-ink font-medium text-[15px]">{file.name}</p>
          ) : (
            <p className="text-muted text-[15px]">
              Drag and drop a CSV file, or click to select
            </p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="flex flex-col gap-2">
            <label className="text-[14px] font-medium text-body-strong">Target Column</label>
            <input
              type="text"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full bg-surface-card rounded-md px-3 py-2 text-ink font-sans text-[14px] border border-hairline-strong focus:border-ink focus:outline-none focus:ring-1 focus:ring-ink transition-all"
              placeholder="e.g. text"
            />
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-body-strong font-medium text-[14px]">
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
              className="w-full accent-ink mt-2"
            />
          </div>
        </div>

        <Button onClick={handleProcess} disabled={!file || loading} className="mt-2 w-full md:w-auto self-start">
          {loading ? "Processing..." : "Process CSV"}
        </Button>
        {error && <p className="text-semantic-error text-[14px]">{error}</p>}
      </FeatureCard>

      {/* Results Preview */}
      {previewLines.length > 0 && downloadUrl && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-hairline pb-2">
            <h3 className="font-serif text-[20px] text-ink">Preview (First 5 Rows)</h3>
            <a
              href={downloadUrl}
              download={`processed_${file?.name || "data.csv"}`}
              className="inline-flex items-center gap-2 text-[14px] font-medium text-ink hover:text-primary-active underline underline-offset-4"
            >
              <Download size={16} />
              Download Full CSV
            </a>
          </div>

          <div className="overflow-x-auto border border-hairline rounded-lg">
            <table className="w-full text-left text-[13px] whitespace-nowrap">
              <thead>
                <tr className="bg-surface-strong border-b border-hairline text-body-strong">
                  {previewLines[0].split(",").map((header, i) => (
                    <th key={i} className="px-4 py-2 font-medium">
                      {header.trim()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-surface-card divide-y divide-hairline text-ink">
                {previewLines.slice(1).map((row, i) => (
                  <tr key={i}>
                    {row.split(",").map((cell, j) => (
                      <td key={j} className="px-4 py-2">
                        {cell.trim().length > 30 ? cell.trim().substring(0, 30) + "..." : cell.trim()}
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
