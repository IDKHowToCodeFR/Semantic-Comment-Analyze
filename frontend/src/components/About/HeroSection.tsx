import React from 'react';
import { motion } from 'framer-motion';

export const HeroSection = () => {
  return (
    <section className="mb-24 mt-12 text-center max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <span className="inline-block px-3 py-1 mb-6 text-[12px] font-sans font-semibold tracking-widest text-muted uppercase border border-hairline rounded-full">
          Platform Architecture
        </span>
        <h1 className="text-5xl md:text-[64px] leading-[1.05] font-serif font-light tracking-[-1.92px] text-ink mb-6">
          Intelligence over <br/> Heuristics.
        </h1>
        <p className="text-[18px] md:text-[20px] text-body leading-relaxed max-w-2xl mx-auto font-sans font-medium">
          A high-performance pipeline abandoning zero-shot classification for a custom Logistic Regression head over <code className="text-ink font-mono text-[15px] bg-surface-strong px-2 py-0.5 rounded">all-MiniLM-L6-v2</code> embeddings, achieving massive throughput.
        </p>
      </motion.div>
    </section>
  );
};
