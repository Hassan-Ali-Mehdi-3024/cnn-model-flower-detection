# Codantrix Labs - Phase 1 Foundation

A premium AI-first software house website built with Next.js, Supabase, and Tailwind CSS.

## 🚀 Features

- **Modern Tech Stack**: Next.js 14+ with App Router, TypeScript, Tailwind CSS
- **Database**: Supabase (PostgreSQL) for backend data management
- **Theme System**: Dark/Light mode with localStorage persistence
- **Animations**: Smooth transitions with Framer Motion
- **Responsive Design**: Mobile-first approach with adaptive layouts
- **Admin Dashboard**: Manage site settings and content
- **SEO Optimized**: Meta tags, Open Graph, and sitemap configured

## 📦 Tech Stack

- **Frontend**: Next.js 14+, React, TypeScript
- **Styling**: Tailwind CSS with CSS variables
- **Database**: Supabase (PostgreSQL)
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Utilities**: clsx, tailwind-merge

## 🎨 Brand Colors

- **Primary Orange**: `#f15a2f`
- **Light Background**: `#fffdf2`
- **Dark Text/Background**: `#1c1e20`
- **Success**: `#10b981`
- **Warning**: `#f59e0b`
- **Error**: `#ef4444`
- **Info**: `#3b82f6`

## 🛠️ Setup Instructions

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Supabase account (free tier works)

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env.local`:

```bash
cp .env.example .env.local
```

Update `.env.local` with your Supabase credentials:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
NEXT_PUBLIC_SITE_NAME=Codantrix Labs
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

### 3. Setup Supabase Database

Go to your Supabase project SQL editor and run the following schema:

```sql
-- Admin Settings
CREATE TABLE admin_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  setting_key TEXT UNIQUE NOT NULL,
  setting_value JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Admin Users (for future auth)
CREATE TABLE admin_users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT DEFAULT 'admin',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Pages (CMS)
CREATE TABLE pages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  meta_description TEXT,
  content JSONB,
  published BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Page Content Blocks
CREATE TABLE page_content (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  page_id UUID REFERENCES pages(id),
  block_type TEXT,
  block_data JSONB,
  order_index INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

-- AI Solutions
CREATE TABLE ai_solutions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  full_content JSONB,
  category TEXT,
  images JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- SaaS Services
CREATE TABLE saas_services (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  category TEXT,
  description TEXT,
  full_content JSONB,
  images JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Products
CREATE TABLE products (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  category TEXT,
  status TEXT DEFAULT 'draft',
  description TEXT,
  long_description JSONB,
  features JSONB,
  pricing_model TEXT,
  images JSONB,
  demo_url TEXT,
  documentation_url TEXT,
  launch_date DATE,
  is_visible BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Case Studies
CREATE TABLE case_studies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  service_type TEXT,
  content JSONB,
  metrics JSONB,
  images JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Blog Posts
CREATE TABLE blog_posts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  content TEXT,
  author TEXT,
  tags JSONB,
  published BOOLEAN DEFAULT FALSE,
  published_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Inquiries
CREATE TABLE inquiries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  inquiry_type TEXT,
  message TEXT,
  status TEXT DEFAULT 'new',
  created_at TIMESTAMP DEFAULT NOW(),
  responded_at TIMESTAMP
);

-- Analytics
CREATE TABLE analytics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  page_slug TEXT,
  user_session TEXT,
  device TEXT,
  referrer TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Seed initial data
INSERT INTO admin_settings (setting_key, setting_value)
VALUES ('products_visible', '{"visible": false}')
ON CONFLICT (setting_key) DO NOTHING;
```

### 4. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
codantrix-labs/
├── app/
│   ├── admin/
│   │   ├── dashboard/
│   │   ├── login/
│   │   ├── settings/
│   │   └── layout.tsx
│   ├── products/
│   ├── layout.tsx
│   ├── page.tsx
│   ├── not-found.tsx
│   ├── error.tsx
│   └── globals.css
├── components/
│   ├── home/
│   │   ├── Hero.tsx
│   │   ├── Synergy.tsx
│   │   ├── ServiceCards.tsx
│   │   ├── CaseStudyCarousel.tsx
│   │   ├── ProductsTeaser.tsx
│   │   ├── AudiencePaths.tsx
│   │   ├── BlogPreview.tsx
│   │   └── FinalCTA.tsx
│   ├── marketing/
│   │   ├── Navbar.tsx
│   │   └── Footer.tsx
│   └── ui/
│       ├── Button.tsx
│       ├── Card.tsx
│       ├── Input.tsx
│       ├── Textarea.tsx
│       ├── Badge.tsx
│       ├── Spinner.tsx
│       └── Modal.tsx
├── lib/
│   ├── supabase.ts
│   ├── theme.tsx
│   ├── auth.ts
│   └── utils.ts
├── public/
│   ├── robots.txt
│   └── sitemap.xml
└── package.json
```

## 🎯 Key Pages

- **Homepage** (`/`) - 8 sections showcasing services, case studies, and CTAs
- **Products** (`/products`) - Products listing (hidden by default)
- **Admin Login** (`/admin/login`) - Authentication page
  - Default credentials: `admin@codantrix.com` / `admin123`
- **Admin Dashboard** (`/admin/dashboard`) - Overview and quick actions
- **Admin Settings** (`/admin/settings`) - Toggle products visibility

## 🎨 Component Library

All UI components are theme-aware and support dark/light modes:

- **Button** - 4 variants (primary, secondary, outline, ghost), 3 sizes
- **Card** - Base card with hover effects
- **Input/Textarea** - Form inputs with error states
- **Badge** - Status badges with color variants
- **Spinner** - Loading indicator
- **Modal** - Dialog component

## 🌙 Theme System

The site supports dark/light mode:

- Toggle using the moon/sun icon in the navbar
- Preference persists in localStorage
- Falls back to system preference
- Smooth transitions between themes

## 🔐 Admin System

Simple client-side authentication for Phase 1:

- Default login: `admin@codantrix.com` / `admin123`
- Session stored in localStorage
- Protected admin routes
- Settings management for products visibility

## 📱 Responsive Design

Breakpoints:

- Mobile: 320px+
- Tablet: 768px+
- Desktop: 1024px+
- Large: 1440px+

## 🚀 Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Import project in Vercel
3. Add environment variables
4. Deploy!

### Other Platforms

The site can be deployed to any platform supporting Next.js:

- Netlify
- AWS Amplify
- Railway
- DigitalOcean App Platform

## 📝 TODO for Phase 2

- [ ] Build out AI Solutions pages
- [ ] Create SaaS Services detail pages
- [ ] Implement Case Studies CMS
- [ ] Add Blog functionality
- [ ] Build Contact form with Supabase
- [ ] Implement proper authentication (NextAuth/Supabase Auth)
- [ ] Add analytics tracking
- [ ] Create sitemap generator
- [ ] Add image optimization

## 🤝 Contributing

This is a Phase 1 foundation. Future phases will expand functionality.

## 📄 License

Proprietary - Codantrix Labs

## 🆘 Support

For issues or questions:

- Email: hello@codantrix.com
- Documentation: /docs (coming soon)

---

Built with ❤️ by Codantrix Labs
