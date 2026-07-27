export default function Logo({ size = 18, strokeWidth = 2 }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: "block" }}
    >
      <defs>
        <linearGradient id="shield-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#00ff26" />
          <stop offset="100%" stopColor="#00ff26" />
        </linearGradient>
      </defs>
      {/* Outer shield structure */}
      <path
        d="M12 2L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-3z"
        stroke="url(#shield-grad)"
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
      />
      {/* Dynamic tech HUD crosshairs inside the shield */}
      <path
        d="M12 7v8M8 11h8"
        stroke="#00ff26"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.6"
      />
      {/* Outer dotted scanning ring */}
      <circle
        cx="12"
        cy="11"
        r="3.5"
        stroke="#ffffff"
        strokeWidth="1"
        strokeDasharray="1.5 1.5"
        opacity="0.8"
      />
      {/* Core active detection sensor */}
      <circle cx="12" cy="11" r="1.5" fill="#00ff26" />
    </svg>
  );
}
