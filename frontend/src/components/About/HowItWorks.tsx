import React from 'react';

export const HowItWorks = () => {
  return (
    <section className="mb-24">
      <h2 className="text-3xl font-serif font-light text-ink mb-12 tracking-tight text-center">
        How It Works
      </h2>
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="bg-surface-card p-6 rounded-2xl border border-hairline">
          <h3 className="text-xl font-medium text-ink mb-2">1. Startup</h3>
          <p className="text-body leading-relaxed">
            FastAPI `lifespan` pre-loads the heavy Hugging Face pipelines and MiniLM models into memory immediately on boot. This ensures zero cold-start latency for the first API request.
          </p>
        </div>
        <div className="bg-surface-card p-6 rounded-2xl border border-hairline">
          <h3 className="text-xl font-medium text-ink mb-2">2. Fine-Tuning</h3>
          <p className="text-body leading-relaxed">
            The custom Intent classifier is a lightweight Logistic Regression head trained purely on dense sentence embeddings, allowing for highly accurate, domain-specific intent detection without the overhead of LLMs.
          </p>
        </div>
        <div className="bg-surface-card p-6 rounded-2xl border border-hairline">
          <h3 className="text-xl font-medium text-ink mb-2">3. Inference</h3>
          <p className="text-body leading-relaxed">
            Incoming requests are offloaded to background threads using `asyncio.to_thread`. This allows the heavy Occlusion Engine calculations to run concurrently without blocking the core FastAPI event loop.
          </p>
        </div>
      </div>
    </section>
  );
};
