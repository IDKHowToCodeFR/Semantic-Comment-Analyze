import React from 'react';
import { HeroSection } from './HeroSection';
import { ArchitectureDiagram } from './ArchitectureDiagram';
import { TechStackGrid } from './TechStackGrid';

export const AboutView = () => {
  return (
    <div className="min-h-full bg-canvas text-ink px-6 py-12 md:px-12 md:py-24 font-sans overflow-x-hidden">
      <HeroSection />
      <ArchitectureDiagram />
      <TechStackGrid />
    </div>
  );
};
