'use client';

import { useEffect } from 'react';
import { AlertCircle, RefreshCw, Home } from 'lucide-react';
import Link from 'next/link';
import Button from '@/components/ui/Button';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Error:', error);
  }, [error]);
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[var(--color-background)] to-[var(--color-card-background)]">
      <div className="text-center px-4 max-w-2xl">
        <div className="w-20 h-20 bg-[var(--color-error)]/10 rounded-full flex items-center justify-center mx-auto mb-6">
          <AlertCircle className="w-10 h-10 text-[var(--color-error)]" />
        </div>
        
        <h1 className="text-4xl font-bold text-[var(--color-foreground)] mb-4">
          Something Went Wrong
        </h1>
        <p className="text-xl text-[var(--color-muted)] mb-8">
          We encountered an unexpected error. Don't worry, our team has been notified 
          and we're working on fixing it.
        </p>
        
        {process.env.NODE_ENV === 'development' && (
          <div className="mb-8 p-4 bg-[var(--color-card-background)] rounded-lg border border-[var(--color-border)] text-left">
            <p className="text-sm text-[var(--color-error)] font-mono">
              {error.message}
            </p>
          </div>
        )}
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            variant="primary"
            size="lg"
            onClick={reset}
            className="flex items-center gap-2"
          >
            <RefreshCw className="w-5 h-5" />
            Try Again
          </Button>
          <Link href="/">
            <Button variant="outline" size="lg" className="flex items-center gap-2">
              <Home className="w-5 h-5" />
              Go Home
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
