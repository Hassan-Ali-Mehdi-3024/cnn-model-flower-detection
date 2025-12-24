import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://placeholder.supabase.co';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'placeholder-key';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Types for database tables
export type AdminSetting = {
  id: string;
  setting_key: string;
  setting_value: any;
  created_at: string;
  updated_at: string;
};

export type Product = {
  id: string;
  slug: string;
  name: string;
  category: string | null;
  status: string;
  description: string | null;
  long_description: any;
  features: any;
  pricing_model: string | null;
  images: any;
  demo_url: string | null;
  documentation_url: string | null;
  launch_date: string | null;
  is_visible: boolean;
  created_at: string;
  updated_at: string;
};

export type CaseStudy = {
  id: string;
  slug: string;
  title: string;
  description: string | null;
  service_type: string | null;
  content: any;
  metrics: any;
  images: any;
  created_at: string;
  updated_at: string;
};

export type BlogPost = {
  id: string;
  slug: string;
  title: string;
  content: string | null;
  author: string | null;
  tags: any;
  published: boolean;
  published_at: string | null;
  created_at: string;
  updated_at: string;
};

export type Inquiry = {
  id: string;
  name: string;
  email: string;
  inquiry_type: string | null;
  message: string | null;
  status: string;
  created_at: string;
  responded_at: string | null;
};
