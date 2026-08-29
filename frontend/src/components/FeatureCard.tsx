import React from "react";

export const FeatureCard = ({
  children,
  className = "",
  title,
}: {
  children: React.ReactNode;
  className?: string;
  title?: string;
}) => {
  return (
    <div
      className={`flex flex-col bg-surface-card text-ink rounded-[16px] border border-hairline-strong shadow-[0_4px_16px_rgba(0,0,0,0.04)] ${className}`}
    >
      {title && (
        <h3 className="font-serif text-[24px] font-[300] text-ink mb-4">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
};
