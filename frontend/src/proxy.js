import { withAuth } from "next-auth/middleware";
import { NextResponse } from "next/server";

export default withAuth(
  function middleware(req) {
    const { pathname } = req.nextUrl;
    const token = req.nextauth.token;

    // Authenticated user hitting /login or /signup → redirect to dashboard
    if ((pathname.startsWith("/login") || pathname.startsWith("/signup")) && token) {
      return NextResponse.redirect(new URL("/dashboard", req.url));
    }

    // Admin-only route guard
    if (pathname.startsWith("/dashboard/users") && token?.role !== "super_admin" && token?.role !== "site_admin") {
      return NextResponse.redirect(new URL("/dashboard", req.url));
    }

    return NextResponse.next();
  },
  {
    // Explicitly pass the secret so withAuth can decrypt the JWT cookie
    secret: process.env.NEXTAUTH_SECRET,
    callbacks: {
      authorized({ token, req }) {
        const { pathname } = req.nextUrl;
        // Allow unauthenticated access to /login and /signup
        if (pathname.startsWith("/login") || pathname.startsWith("/signup")) return true;
        // All /dashboard/* require a token
        if (pathname.startsWith("/dashboard")) return !!token;
        return true;
      },
    },
  }
);

export const config = {
  matcher: ["/dashboard/:path*", "/login", "/signup"],
};
