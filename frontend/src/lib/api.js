const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Base fetch wrapper — attaches Bearer token from session.
 * @param {string} path  - API path e.g. "/api/violations"
 * @param {object} options - fetch options (method, body, etc.)
 * @param {object|null} session - NextAuth session object
 */
export async function apiFetch(path, options = {}, session = null) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(session?.access_token
        ? { Authorization: `Bearer ${session.access_token}` }
        : {}),
      ...options.headers,
    },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }

  // Handle empty responses (e.g. 204 No Content)
  if (res.status === 204) {
    return null;
  }
  
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res;
}

export { API_BASE };
