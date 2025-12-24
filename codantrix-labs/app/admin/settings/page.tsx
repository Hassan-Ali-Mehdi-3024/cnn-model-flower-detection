'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Save, Check } from 'lucide-react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Spinner from '@/components/ui/Spinner';
import { supabase } from '@/lib/supabase';

export default function AdminSettings() {
  const [productsVisible, setProductsVisible] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  
  useEffect(() => {
    fetchSettings();
  }, []);
  
  const fetchSettings = async () => {
    try {
      const { data, error } = await supabase
        .from('admin_settings')
        .select('setting_value')
        .eq('setting_key', 'products_visible')
        .single();
      
      if (error) {
        console.log('No existing settings found');
        setProductsVisible(false);
      } else {
        setProductsVisible(data?.setting_value?.visible || false);
      }
    } catch (err) {
      console.error('Error fetching settings:', err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    
    try {
      // Check if setting exists
      const { data: existing } = await supabase
        .from('admin_settings')
        .select('id')
        .eq('setting_key', 'products_visible')
        .single();
      
      if (existing) {
        // Update existing setting
        const { error } = await supabase
          .from('admin_settings')
          .update({
            setting_value: { visible: productsVisible },
            updated_at: new Date().toISOString()
          })
          .eq('setting_key', 'products_visible');
        
        if (error) throw error;
      } else {
        // Insert new setting
        const { error } = await supabase
          .from('admin_settings')
          .insert({
            setting_key: 'products_visible',
            setting_value: { visible: productsVisible }
          });
        
        if (error) throw error;
      }
      
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Error saving settings:', err);
      alert('Error saving settings. Please check your Supabase configuration.');
    } finally {
      setSaving(false);
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }
  
  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl font-bold mb-2 text-[var(--color-foreground)]">
          Settings
        </h1>
        <p className="text-[var(--color-muted)] mb-8">
          Manage your site settings and preferences
        </p>
        
        <Card className="p-8">
          <h2 className="text-2xl font-bold mb-6 text-[var(--color-foreground)]">
            Products Section
          </h2>
          
          <div className="space-y-6">
            <div className="flex items-center justify-between p-4 bg-[var(--color-background)] rounded-lg">
              <div>
                <h3 className="font-semibold text-[var(--color-foreground)] mb-1">
                  Products Visibility
                </h3>
                <p className="text-sm text-[var(--color-muted)]">
                  Toggle whether products are visible on the homepage and products page
                </p>
              </div>
              
              <button
                onClick={() => setProductsVisible(!productsVisible)}
                className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                  productsVisible ? 'bg-[var(--color-primary)]' : 'bg-[var(--color-muted)]'
                }`}
              >
                <span
                  className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                    productsVisible ? 'translate-x-7' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
            
            <div className="p-4 bg-[var(--color-info)]/10 border border-[var(--color-info)] rounded-lg">
              <p className="text-sm text-[var(--color-info)]">
                <strong>Note:</strong> When products are hidden, visitors will see a "Coming Soon" message 
                with a waitlist signup form. When visible, the products grid will be displayed.
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              <Button
                variant="primary"
                size="lg"
                onClick={handleSave}
                loading={saving}
                className="flex items-center gap-2"
              >
                {saved ? (
                  <>
                    <Check className="w-4 h-4" />
                    Saved!
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" />
                    Save Changes
                  </>
                )}
              </Button>
              
              <span className="text-sm text-[var(--color-muted)]">
                Current status: 
                <strong className={productsVisible ? 'text-[var(--color-success)]' : 'text-[var(--color-warning)]'}>
                  {' '}{productsVisible ? 'Visible' : 'Hidden'}
                </strong>
              </span>
            </div>
          </div>
        </Card>
        
        {/* Database Configuration Info */}
        <Card className="p-6 mt-6 bg-[var(--color-warning)]/10 border border-[var(--color-warning)]">
          <h3 className="font-semibold text-[var(--color-foreground)] mb-2">
            Database Configuration
          </h3>
          <p className="text-sm text-[var(--color-muted)] mb-4">
            Make sure you have configured your Supabase credentials in the .env.local file 
            and created the required database tables.
          </p>
          <details className="text-sm">
            <summary className="cursor-pointer text-[var(--color-primary)] font-medium">
              View required environment variables
            </summary>
            <div className="mt-2 p-3 bg-[var(--color-card-background)] rounded border border-[var(--color-border)] font-mono text-xs">
              <p>NEXT_PUBLIC_SUPABASE_URL=your_supabase_url</p>
              <p>NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key</p>
              <p>SUPABASE_SERVICE_ROLE_KEY=your_service_role_key</p>
            </div>
          </details>
        </Card>
      </motion.div>
    </div>
  );
}
