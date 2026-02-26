"use client";

import React, { useMemo } from 'react';
import { ReactFlow, Background, Controls } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { CustomNode } from './CustomNode';

export const ArchitectureDiagram = () => {
  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);

  const nodes = [
    { id: '1', type: 'custom', position: { x: 400, y: 0 }, data: { label: 'Client Request', icon: 'file' } },
    { id: '2', type: 'custom', position: { x: 400, y: 120 }, data: { label: 'FastAPI Server', icon: 'box', subline: 'REST API' } },
    
    // NLP Pipeline Subaxis
    { id: '3', type: 'custom', position: { x: 400, y: 260 }, data: { label: 'NLP Engine', icon: 'cpu' } },
    { id: '4', type: 'custom', position: { x: 150, y: 400 }, data: { label: 'Embeddings', icon: 'brain', subline: 'MiniLM-L6' } },
    { id: '5', type: 'custom', position: { x: 150, y: 540 }, data: { label: 'Intent & Confidence', icon: 'activity', subline: 'Custom Head' } },
    { id: '6', type: 'custom', position: { x: 400, y: 400 }, data: { label: 'Sentiment Pipeline', icon: 'brain', subline: 'Hugging Face' } },
    { id: '7', type: 'custom', position: { x: 650, y: 400 }, data: { label: 'NER Pipeline', icon: 'brain', subline: 'Hugging Face' } },
    { id: '8', type: 'custom', position: { x: 900, y: 400 }, data: { label: 'Occlusion Algorithm', icon: 'activity' } },
    
    // Business Logic
    { id: '9', type: 'custom', position: { x: 400, y: 540 }, data: { label: 'Sentiment Array', icon: 'activity' } },
    { id: '10', type: 'custom', position: { x: 650, y: 540 }, data: { label: 'Entities', icon: 'activity' } },
    
    { id: '11', type: 'custom', position: { x: 400, y: 680 }, data: { label: 'Heuristics Evaluator', icon: 'cpu' } },
    { id: '12', type: 'custom', position: { x: 400, y: 820 }, data: { label: 'Urgency, Tone, Action', icon: 'activity' } },
    
    // Response
    { id: '13', type: 'custom', position: { x: 400, y: 960 }, data: { label: 'Aggregated JSON', icon: 'database' } },
  ];

  const edges = [
    { id: 'e1-2', source: '1', target: '2', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e2-3', source: '2', target: '3', animated: true, style: { stroke: '#a8a29e' } },
    
    // From NLP Engine
    { id: 'e3-4', source: '3', target: '4', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e3-6', source: '3', target: '6', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e3-7', source: '3', target: '7', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e3-8', source: '3', target: '8', animated: true, style: { stroke: '#a8a29e' } },
    
    // Pipeline intermediate steps
    { id: 'e4-5', source: '4', target: '5', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e6-9', source: '6', target: '9', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e7-10', source: '7', target: '10', animated: true, style: { stroke: '#a8a29e' } },
    
    // To Heuristics
    { id: 'e5-11', source: '5', target: '11', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e9-11', source: '9', target: '11', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e11-12', source: '11', target: '12', animated: true, style: { stroke: '#a8a29e' } },
    
    // To Response
    { id: 'e12-13', source: '12', target: '13', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e10-13', source: '10', target: '13', animated: true, style: { stroke: '#a8a29e' } },
    { id: 'e8-13', source: '8', target: '13', animated: true, style: { stroke: '#a8a29e' } },
  ];

  return (
    <section className="mb-24">
      <h2 className="text-3xl font-serif font-light text-ink mb-12 tracking-tight text-center">Data Flow Architecture</h2>
      <div className="w-full h-[600px] border border-hairline rounded-2xl bg-canvas-soft overflow-hidden">
        <ReactFlow 
          nodes={nodes} 
          edges={edges} 
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.2}
          maxZoom={1.5}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#e7e5e4" gap={24} />
          <Controls className="!bg-surface-card !border-hairline !text-ink" />
        </ReactFlow>
      </div>
    </section>
  );
};
