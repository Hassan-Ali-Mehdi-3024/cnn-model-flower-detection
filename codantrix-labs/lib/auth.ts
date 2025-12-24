'use client';

// Simple client-side auth simulation for Phase 1
// In production, replace with proper authentication (NextAuth, Supabase Auth, etc.)

const ADMIN_CREDENTIALS = {
  email: 'admin@codantrix.com',
  password: 'admin123', // In production, use proper hashing
};

export function login(email: string, password: string): boolean {
  if (email === ADMIN_CREDENTIALS.email && password === ADMIN_CREDENTIALS.password) {
    localStorage.setItem('admin_authenticated', 'true');
    localStorage.setItem('admin_email', email);
    return true;
  }
  return false;
}

export function logout(): void {
  localStorage.removeItem('admin_authenticated');
  localStorage.removeItem('admin_email');
}

export function isAuthenticated(): boolean {
  if (typeof window === 'undefined') return false;
  return localStorage.getItem('admin_authenticated') === 'true';
}

export function getAdminEmail(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('admin_email');
}

export function requireAuth(callback: () => void): void {
  if (!isAuthenticated()) {
    window.location.href = '/admin/login';
  } else {
    callback();
  }
}
