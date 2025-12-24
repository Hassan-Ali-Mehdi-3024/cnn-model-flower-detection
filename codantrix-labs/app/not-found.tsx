'use client';

import Link from 'next/link';
import { Home, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[var(--color-background)] to-[var(--color-card-background)] p-4">
      <div className="text-center">
        <h1 className="text-9xl font-bold text-[var(--color-primary)] mb-4">404</h1>
        <h2 className="text-4xl font-bold text-[var(--color-foreground)] mb-4">
          Page Not Found
        </h2>
        <p className="text-xl text-[var(--color-muted)] mb-8 max-w-md mx-auto">
          Sorry, we couldn't find the page you're looking for. It might have been moved or deleted.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            href="/"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 text-lg rounded-lg font-medium transition-all duration-200 bg-[var(--color-primary)] text-white hover:opacity-90"
          >
            <Home className="w-5 h-5" />
            Go Home
          </Link>
          <button
            onClick={() => window.history.back()}
            className="inline-flex items-center justify-center gap-2 px-6 py-3 text-lg rounded-lg font-medium transition-all duration-200 border-2 border-[var(--color-primary)] text-[var(--color-primary)] hover:bg-[var(--color-primary)] hover:text-white"
          >
            <ArrowLeft className="w-5 h-5" />
            Go Back
          </button>
        </div>
        
        <div className="mt-12">
          <p className="text-sm text-[var(--color-muted)] mb-2">Quick Links</p>
          <div className="flex flex-wrap gap-4 justify-center text-sm">
            <Link href="/ai-solutions" className="text-[var(--color-primary)] hover:underline">
              AI Solutions
            </Link>
            <Link href="/services" className="text-[var(--color-primary)] hover:underline">
              Services
            </Link>
            <Link href="/products" className="text-[var(--color-primary)] hover:underline">
              Products
            </Link>
            <Link href="/contact" className="text-[var(--color-primary)] hover:underline">
              Contact
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
