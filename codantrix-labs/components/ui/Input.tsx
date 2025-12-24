import { InputHTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  error?: boolean;
  label?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, error = false, label, type = 'text', ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium mb-2 text-[var(--color-foreground)]">
            {label}
          </label>
        )}
        <input
          type={type}
          ref={ref}
          className={cn(
            'w-full px-4 py-2 rounded-lg border bg-[var(--color-card-background)] text-[var(--color-foreground)]',
            'focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent',
            'transition-all duration-200',
            error ? 'border-[var(--color-error)]' : 'border-[var(--color-border)]',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            className
          )}
          {...props}
        />
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
