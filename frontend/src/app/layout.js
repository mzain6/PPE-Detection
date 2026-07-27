import "./globals.css";
import { getServerSession } from "next-auth";
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
  const session = await getServerSession();
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
