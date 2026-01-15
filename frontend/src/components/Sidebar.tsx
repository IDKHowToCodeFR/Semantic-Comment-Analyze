import React from "react";
import { LayoutDashboard, Database } from "lucide-react";

export type ViewType = "single" | "batch";

export const Sidebar = ({
  activeView,
  onViewChange,
}: {
  activeView: ViewType;
  onViewChange: (view: ViewType) => void;
}) => {
  return (
    <aside className="w-64 border-r border-hairline bg-surface-card flex flex-col shrink-0 h-screen overflow-y-auto">
      <div className="p-6 border-b border-hairline">
        <h1 className="font-serif text-[20px] font-medium text-ink tracking-tight">
          Semantic Analysis
        </h1>
        <p className="text-[13px] text-muted mt-1 font-sans">
          NLP Workspace
        </p>
      </div>

      <nav className="flex-1 p-4 flex flex-col gap-2">
        <button
          onClick={() => onViewChange("single")}
          className={`flex items-center gap-3 px-4 py-2.5 rounded-md text-[14px] font-sans font-medium transition-colors ${
            activeView === "single"
              ? "bg-surface-strong text-ink"
              : "text-body hover:bg-canvas hover:text-ink"
          }`}
        >
          <LayoutDashboard size={18} />
          Single Analysis
        </button>

        <button
          onClick={() => onViewChange("batch")}
          className={`flex items-center gap-3 px-4 py-2.5 rounded-md text-[14px] font-sans font-medium transition-colors ${
            activeView === "batch"
              ? "bg-surface-strong text-ink"
              : "text-body hover:bg-canvas hover:text-ink"
          }`}
        >
          <Database size={18} />
          Batch Processing
        </button>
      </nav>
    </aside>
  );
};
