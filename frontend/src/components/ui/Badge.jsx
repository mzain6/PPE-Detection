export default function Badge({ variant = "neutral", children, style }) {
  return (
    <span className={`badge badge-${variant}`} style={style}>
      {children}
    </span>
  );
}

/**
 * Maps violation type string to badge variant
 */
export function ViolationBadge({ type }) {
  const map = {
    no_helmet: { variant: "danger",  label: "No Helmet" },
    no_vest:   { variant: "warning", label: "No Vest"   },
    no_gloves: { variant: "warning", label: "No Gloves" },
    no_both:   { variant: "danger",  label: "No PPE"    },
  };
  const { variant, label } = map[type] || { variant: "neutral", label: type };
  return <Badge variant={variant}>{label}</Badge>;
}

/**
 * Maps user role to badge variant
 */
export function RoleBadge({ role }) {
  const map = {
    super_admin:    { variant: "primary", label: "Super Admin"    },
    site_admin:     { variant: "info",    label: "Site Admin"     },
    safety_officer: { variant: "info",    label: "Safety Officer" },
    viewer:         { variant: "neutral", label: "Viewer"         },
  };
  const { variant, label } = map[role] || { variant: "neutral", label: role };
  return <Badge variant={variant}>{label}</Badge>;
}
