import type { Metadata, Viewport } from "next";
import "./globals.css";
import { ThemeProvider } from "@/lib/theme";
import Navbar from "@/components/marketing/Navbar";
import Footer from "@/components/marketing/Footer";

export const metadata: Metadata = {
  title: "Codantrix Labs | AI Solutions & SaaS Services",
  description: "Custom AI solutions and enterprise SaaS services for modern businesses. Transform your business with cutting-edge technology.",
  keywords: ["AI solutions", "SaaS services", "custom software", "machine learning", "web development"],
  authors: [{ name: "Codantrix Labs" }],
  openGraph: {
    title: "Codantrix Labs | AI Solutions & SaaS Services",
    description: "Custom AI solutions and enterprise SaaS services for modern businesses",
    url: "https://codantrix.com",
    siteName: "Codantrix Labs",
    images: [
      {
        url: "/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "Codantrix Labs",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Codantrix Labs | AI Solutions & SaaS Services",
    description: "Custom AI solutions and enterprise SaaS services for modern businesses",
    images: ["/og-image.jpg"],
  },
  icons: {
    icon: "/favicon.ico",
  },
};

export const viewport: Viewport = {
  themeColor: "#f15a2f",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <ThemeProvider>
          <Navbar />
          <main className="min-h-screen pt-16">
            {children}
          </main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  );
}
