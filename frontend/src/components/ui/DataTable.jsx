import { ChevronLeft, ChevronRight, Inbox } from "lucide-react";

function SkeletonRow({ cols }) {
  return (
    <tr>
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i}>
          <span className="skeleton" style={{ display: "block", height: 14, width: "70%", borderRadius: 4 }} />
        </td>
      ))}
    </tr>
  );
}

export default function DataTable({
  columns,
  data,
  loading,
  emptyMessage = "No data found",
  currentPage = 1,
  totalPages = 1,
  total = 0,
  pageSize = 25,
  onPageChange,
}) {
  const startRow = (currentPage - 1) * pageSize + 1;
  const endRow   = Math.min(currentPage * pageSize, total);

  return (
    <div className="table-wrap">
      <div style={{ overflowX: "auto" }}>
        <table className="table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key}>{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading
              ? Array.from({ length: 5 }).map((_, i) => (
                  <SkeletonRow key={i} cols={columns.length} />
                ))
              : data?.length === 0
              ? (
                <tr>
                  <td colSpan={columns.length}>
                    <div className="empty-state">
                      <Inbox size={40} className="empty-state-icon" />
                      <span className="empty-state-title">{emptyMessage}</span>
                    </div>
                  </td>
                </tr>
              )
              : data?.map((row, rowIdx) => (
                <tr key={row.id || rowIdx}>
                  {columns.map((col) => (
                    <td key={col.key}>
                      {col.render ? col.render(row[col.key], row) : row[col.key]}
                    </td>
                  ))}
                </tr>
              ))
            }
          </tbody>
        </table>
      </div>

      {onPageChange && total > 0 && (
        <div className="pagination">
          <span>
            Showing {startRow}–{endRow} of {total} results
          </span>
          <div className="pagination-controls">
            <button
              className="page-btn"
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage <= 1}
              aria-label="Previous page"
            >
              <ChevronLeft size={14} />
            </button>
            <span style={{ fontSize: "0.8rem", color: "var(--color-text-secondary)", padding: "0 0.5rem" }}>
              {currentPage} / {totalPages}
            </span>
            <button
              className="page-btn"
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage >= totalPages}
              aria-label="Next page"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
