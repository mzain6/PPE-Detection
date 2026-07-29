import "./globals.css";
import { getServerSession } from "next-auth";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import SessionProvider from "@/components/SessionProvider";
import { Inter } from "next/font/google";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata = {
  title: "SafeSite AI — PPE Detection Dashboard",
  description: "Real-time PPE violation detection and monitoring dashboard for construction sites.",
};

export default async function RootLayout({ children }) {
  let session = null;
  try {
    session = await getServerSession(authOptions);
  } catch {
    // Stale or invalid session cookie (e.g. encrypted with a different secret).
    // Treat as unauthenticated — the browser will clear the cookie on next signIn.
    session = null;
  }
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <SessionProvider session={session}>
          {children}
        </SessionProvider>
      </body>
    </html>
  );
}
