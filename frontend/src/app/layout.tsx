import type { Metadata } from "next";
import { Inter, EB_Garamond } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const garamond = EB_Garamond({ subsets: ["latin"], variable: "--font-garamond", weight: "400" }); // 300 is unavailable in standard EB_Garamond via Google Fonts without specific handling, falling back to 400

export const metadata: Metadata = {
  title: "Semantic Analysis",
  description: "Advanced semantic analysis for comments and text data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${garamond.variable}`}>
        {children}
      </body>
    </html>
  );
}
