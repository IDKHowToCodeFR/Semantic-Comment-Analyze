"use client";

import React, { useState } from "react";
import { Sidebar, ViewType } from "@/components/Sidebar";
import { SingleAnalysisView } from "@/components/SingleAnalysisView";
import { BatchProcessingView } from "@/components/BatchProcessingView";
import { AboutView } from "@/components/About/AboutView";

export default function Home() {
  const [activeView, setActiveView] = useState<ViewType>("single");

  return (
    <div className="flex h-screen bg-canvas overflow-hidden text-ink">
      <Sidebar activeView={activeView} onViewChange={setActiveView} />
      <main className="flex-1 h-full overflow-y-auto">
        {activeView === "single" ? <SingleAnalysisView /> : activeView === "batch" ? <BatchProcessingView /> : <AboutView />}
      </main>
    </div>
  );
}
