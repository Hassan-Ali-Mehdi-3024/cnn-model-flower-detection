'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Send } from 'lucide-react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Textarea from '@/components/ui/Textarea';

export default function FinalCTA() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });
  const [submitted, setSubmitted] = useState(false);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement form submission to Supabase
    console.log('Form submitted:', formData);
    setSubmitted(true);
    setTimeout(() => {
      setSubmitted(false);
      setFormData({ name: '', email: '', message: '' });
    }, 3000);
  };
  
  return (
    <section className="py-20 bg-gradient-to-br from-[var(--color-primary)]/10 via-[var(--color-background)] to-[var(--color-info)]/10">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="max-w-3xl mx-auto"
        >
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-5xl font-bold mb-4 text-[var(--color-foreground)]">
              Let's Build <span className="text-[var(--color-primary)]">Together</span>
            </h2>
            <p className="text-xl text-[var(--color-muted)]">
              Ready to transform your business with AI? Get in touch with our team.
            </p>
          </div>
          
          <Card className="p-8">
            {submitted ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center py-12"
              >
                <div className="w-20 h-20 bg-[var(--color-success)]/20 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Send className="w-10 h-10 text-[var(--color-success)]" />
                </div>
                <h3 className="text-2xl font-bold mb-2 text-[var(--color-foreground)]">
                  Message Sent!
                </h3>
                <p className="text-[var(--color-muted)]">
                  We'll get back to you within 24 hours.
                </p>
              </motion.div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Input
                    label="Name"
                    placeholder="Your name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    required
                  />
                  <Input
                    label="Email"
                    type="email"
                    placeholder="your@email.com"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    required
                  />
                </div>
                
                <Textarea
                  label="Message"
                  placeholder="Tell us about your project..."
                  value={formData.message}
                  onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                  required
                  rows={5}
                />
                
                <Button type="submit" variant="primary" size="lg" className="w-full">
                  Send Message
                </Button>
              </form>
            )}
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
