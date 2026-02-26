import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { Box, Cpu, BrainCircuit, Activity, Database, FileText, LucideIcon } from 'lucide-react';

const iconMap: Record<string, LucideIcon> = {
  'box': Box,
  'cpu': Cpu,
  'brain': BrainCircuit,
  'activity': Activity,
  'database': Database,
  'file': FileText
};

interface CustomNodeData {
  label: string;
  icon?: string;
  subline?: string;
}

export const CustomNode = ({ data }: { data: CustomNodeData }) => {
  const IconComponent = iconMap[data.icon || 'box'];

  return (
    <div className="bg-surface-card border border-hairline rounded-xl p-4 shadow-sm min-w-[200px] text-center font-sans">
      <Handle type="target" position={Position.Top} className="!bg-hairline-strong !w-2 !h-2" />
      
      <div className="w-10 h-10 mx-auto bg-canvas-soft border border-hairline rounded-full flex items-center justify-center text-ink mb-3">
        <IconComponent size={20} />
      </div>
      <div className="font-medium text-[14px] text-ink mb-1">{data.label}</div>
      {data.subline && (
        <div className="text-[12px] text-muted">{data.subline}</div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-hairline-strong !w-2 !h-2" />
    </div>
  );
};
