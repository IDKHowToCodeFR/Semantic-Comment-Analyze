"use client";

import React from 'react';
import Image from 'next/image';

export const ArchitectureDiagram = () => {
  return (
    <section className="mb-24">
      <h2 className="text-3xl font-serif font-light text-ink mb-12 tracking-tight text-center">
        Data Flow Architecture
      </h2>
      <div className="relative w-full h-[600px] border border-hairline rounded-2xl bg-canvas-soft overflow-hidden p-8">
        <Image 
          src="/arch.svg" 
          alt="Architecture Diagram" 
          fill
          className="object-contain"
        />
      </div>
    </section>
  );
};
