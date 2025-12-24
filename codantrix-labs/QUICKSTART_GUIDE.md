# Codantrix Labs - Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Navigate to Project
```bash
cd /home/engine/project/codantrix-labs
```

### Step 2: Install Dependencies (if not already done)
```bash
npm install
```

### Step 3: Start Development Server
```bash
npm run dev
```

The site will be available at **http://localhost:3000**

## 🎨 What You'll See

### Homepage (/)
- **Hero Section**: AI demo with neural network visualization
- **8 Complete Sections**: Services, case studies, products teaser, blog preview, etc.
- **Theme Toggle**: Click the sun/moon icon in navbar to switch themes
- **Fully Responsive**: Try resizing your browser

### Admin Panel (/admin/login)
**Credentials:**
- Email: `admin@codantrix.com`
- Password: `admin123`

**Features:**
- Dashboard with stats
- Settings to toggle products visibility
- Logout functionality

### Products Page (/products)
- Shows "Coming Soon" by default
- After toggling visibility in admin settings, refresh to see changes

## 🗄️ Database Setup (Optional for Phase 1)

If you want to test the admin settings functionality:

1. **Create Supabase Account**: https://supabase.com (free)

2. **Get Your Credentials**:
   - Project URL: `https://[project-id].supabase.co`
   - Anon Key: Found in Project Settings → API

3. **Update .env.local**:
   ```bash
   NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your-actual-anon-key
   ```

4. **Run SQL Schema**:
   - Go to Supabase SQL Editor
   - Copy content from `supabase-schema.sql`
   - Execute it
   - Verify tables created

5. **Restart Dev Server**:
   ```bash
   npm run dev
   ```

## 📝 Common Tasks

### Change Theme
- Click sun/moon icon in navbar
- Theme persists in localStorage

### Access Admin
1. Go to `/admin/login`
2. Enter credentials (see above)
3. Navigate to Settings
4. Toggle products visibility
5. Visit `/products` to see change

### Build for Production
```bash
npm run build
npm run start
```

### Check TypeScript
```bash
npx tsc --noEmit
```

## 🎯 Key Features to Test

- ✅ Homepage loads all 8 sections
- ✅ Dark/Light theme toggle
- ✅ Navbar dropdowns (hover on desktop)
- ✅ Mobile menu (resize to mobile)
- ✅ Admin login & dashboard
- ✅ Products visibility toggle
- ✅ Responsive design (all breakpoints)
- ✅ Smooth animations

## 🔧 Troubleshooting

### Build Fails
- Make sure `.env.local` has valid Supabase URLs (placeholder values work too)
- Run `npm install` again
- Clear `.next` folder: `rm -rf .next`

### Theme Not Working
- Check browser localStorage
- Try hard refresh (Cmd/Ctrl + Shift + R)
- Clear site data in browser

### Admin Login Not Working
- Credentials are hardcoded: admin@codantrix.com / admin123
- Check browser console for errors

### Supabase Errors
- Verify credentials in `.env.local`
- Check if tables exist in Supabase dashboard
- Ensure anon key has correct permissions

## 📚 File Structure

```
codantrix-labs/
├── app/
│   ├── page.tsx              # Homepage (8 sections)
│   ├── layout.tsx            # Root layout
│   ├── admin/                # Admin pages
│   │   ├── login/
│   │   ├── dashboard/
│   │   └── settings/
│   └── products/             # Products page
├── components/
│   ├── home/                 # Homepage sections
│   ├── marketing/            # Navbar, Footer
│   └── ui/                   # Reusable components
└── lib/
    ├── supabase.ts           # Database client
    ├── theme.tsx             # Theme system
    └── auth.ts               # Auth helpers
```

## 🎨 Customization

### Change Colors
Edit `app/globals.css`:
```css
:root {
  --color-primary: #f15a2f;  /* Change this */
  /* ... */
}
```

### Add New Page
1. Create `app/your-page/page.tsx`
2. Add route to Navbar in `components/marketing/Navbar.tsx`

### Modify Homepage
Edit sections in `components/home/`

## 🚀 Ready for Development!

You're all set! The complete Phase 1 foundation is ready.

**Next**: Review PHASE1_SUMMARY.md for detailed information about what's been built.

---

Need help? Check README.md for comprehensive documentation.
