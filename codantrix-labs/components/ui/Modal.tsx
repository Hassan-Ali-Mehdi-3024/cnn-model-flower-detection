'use client';

import { useEffect } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export default function Modal({ isOpen, onClose, title, children, className }: ModalProps) {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal Content */}
      <div className={cn(
        'relative z-10 w-full max-w-lg mx-4 bg-[var(--color-card-background)] rounded-lg shadow-xl',
        className
      )}>
        {/* Header */}
        {title && (
          <div className="flex items-center justify-between p-6 border-b border-[var(--color-border)]">
            <h2 className="text-xl font-semibold text-[var(--color-foreground)]">{title}</h2>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-[var(--color-background)] transition-colors"
            >
              <X className="w-5 h-5 text-[var(--color-foreground)]" />
            </button>
          </div>
        )}
        
        {/* Body */}
        <div className="p-6">
          {children}
        </div>
      </div>
    </div>
  );
}
