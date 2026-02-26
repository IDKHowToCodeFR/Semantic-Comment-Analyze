import React from 'react';
import { motion, Variants } from 'framer-motion';

export const TechStackGrid = () => {
  const container: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item: Variants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } }
  };

  return (
    <section className="mb-24">
      <h2 className="text-3xl font-serif font-light text-ink mb-12 tracking-tight text-center">System Components</h2>
      <motion.div 
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-50px" }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto"
      >
        <motion.div variants={item} className="bg-surface-card p-8 rounded-2xl border border-hairline hover:border-hairline-strong transition-colors">
          <h3 className="text-[18px] font-sans font-medium text-ink mb-3">Custom Intent Head</h3>
          <p className="text-[15px] font-sans text-body leading-relaxed">
            A lightweight Logistic Regression classifier trained directly on dense embeddings. Avoids the high latency of zero-shot models while maintaining domain-specific accuracy.
          </p>
        </motion.div>

        <motion.div variants={item} className="bg-surface-card p-8 rounded-2xl border border-hairline hover:border-hairline-strong transition-colors">
          <h3 className="text-[18px] font-sans font-medium text-ink mb-3">Occlusion Engine</h3>
          <p className="text-[15px] font-sans text-body leading-relaxed">
            Eliminates the AI black box. Iteratively masks words and measures confidence delta to provide exact word-level contributions for every prediction.
          </p>
        </motion.div>

        <motion.div variants={item} className="bg-surface-card p-8 rounded-2xl border border-hairline hover:border-hairline-strong transition-colors">
          <h3 className="text-[18px] font-sans font-medium text-ink mb-3">Business Heuristics</h3>
          <p className="text-[15px] font-sans text-body leading-relaxed">
            Deterministic rule layers (`evaluation.py`) that translate raw ML probabilities into actionable business metrics: Urgency, Tone, and Recommended Actions.
          </p>
        </motion.div>

        <motion.div variants={item} className="bg-surface-card p-8 rounded-2xl border border-hairline hover:border-hairline-strong transition-colors md:col-span-2 lg:col-span-3">
          <h3 className="text-[18px] font-sans font-medium text-ink mb-4">File Structure</h3>
          <div className="bg-canvas-soft rounded-xl p-4 overflow-x-auto border border-hairline-soft">
            <pre className="text-[13px] font-mono text-muted leading-[1.6]">
{`src/
├── api/server.py          # FastAPI application and endpoint routing
├── engine/
│   ├── nlp_engine.py      # Transformer models, occlusion, and inference logic
│   ├── train_intent.py    # Train custom Intent Classifier head
│   ├── train_sentiment.py # Fine-tune HF RoBERTa Sentiment Model
│   └── evaluation.py      # Business heuristics (Urgency, Tone, Actions)
└── data/
    └── data_handler.py    # High-throughput CSV/Batch processing generator`}
            </pre>
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
};
